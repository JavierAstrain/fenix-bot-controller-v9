
from __future__ import annotations
import re, unicodedata, datetime as dt
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import pandas as pd
import yaml, json

# -------------------- utils --------------------
def _norm(s:str)->str:
    s = str(s or "").replace("\u00A0"," ").strip()
    s = ''.join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+"," ", s).lower()
    return s

MESES = {"enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,"julio":7,"agosto":8,
         "septiembre":9,"setiembre":9,"octubre":10,"noviembre":11,"diciembre":12}

@dataclass
class Plan:
    metric: str
    dims: List[str]=field(default_factory=list)
    op: str="sum"
    filters: List[Dict[str,str]]=field(default_factory=list)

# -------------------- dictionary --------------------
def build_dictionary(dict_df: Optional[pd.DataFrame], json_static: Optional[list]) -> dict:
    out = {"map": {}}
    # static list
    if isinstance(json_static, list):
        for rec in json_static:
            hoja = str(rec.get("Hoja","")).strip().upper()
            campo= str(rec.get("Campo","")).strip()
            rol  = str(rec.get("Rol","")).strip().lower() or "text"
            alias= str(rec.get("Alias / Sinónimos","") or rec.get("Alias","")).strip()
            out["map"].setdefault(hoja, {"roles":{}, "aliases":{}})
            if campo: out["map"][hoja]["roles"][campo]=rol
            if alias:
                for a in re.split(r"[|,;/]", alias):
                    a=a.strip()
                    if a: out["map"][hoja]["aliases"][_norm(a)] = campo
    # sheet dictionary has priority
    if isinstance(dict_df, pd.DataFrame) and not dict_df.empty:
        cols = { _norm(c): c for c in dict_df.columns }
        c_sheet = cols.get("hoja") or cols.get("sheet")
        c_field = cols.get("campo") or cols.get("columna") or cols.get("nombre")
        c_role  = cols.get("rol") or cols.get("role")
        c_alias = cols.get("alias") or cols.get("alias / sinónimos") or cols.get("alias / sinonimos")
        for _, r in dict_df.iterrows():
            h = str(r.get(c_sheet,"MODELO_BOT")).strip().upper()
            field = str(r.get(c_field,"")).strip()
            role  = str(r.get(c_role,"")).strip().lower() or "text"
            alias = str(r.get(c_alias,"")).strip() if c_alias in dict_df.columns else ""
            out["map"].setdefault(h, {"roles":{}, "aliases":{}})
            if field: out["map"][h]["roles"][field]=role
            if alias:
                for a in re.split(r"[|,;/]", alias):
                    a=a.strip()
                    if a: out["map"][h]["aliases"][_norm(a)] = field
    return out

def resolve_col(df: pd.DataFrame, dic_map: dict, hoja: str, name_or_alias: str, prefer_role: Optional[str]=None) -> Optional[str]:
    if df is None or df.empty: return None
    info = (dic_map.get("map") or {}).get(hoja) or {}
    roles = info.get("roles", {})
    aliases = info.get("aliases", {})
    nn = _norm(name_or_alias)
    # alias exacto
    if nn in aliases:
        real = aliases[nn]
        if real in df.columns: return real
    # nombre exacto (normalizado)
    for c in df.columns:
        if _norm(c)==nn: return c
    # por rol sugerido
    if prefer_role:
        for c,r in roles.items():
            if r==prefer_role and c in df.columns: return c
    # contiene
    for c in df.columns:
        if nn in _norm(c): return c
    # por rol genérico buscando coincidencias
    if prefer_role:
        for c,r in roles.items():
            if r==prefer_role:
                for c2 in df.columns:
                    if _norm(c2)==_norm(c): return c2
    return None

def _pick_date(df, dic_map, hoja):
    info = (dic_map.get("map") or {}).get(hoja) or {}
    roles = info.get("roles", {})
    for c,r in roles.items():
        if r=="date" and c in df.columns: return c
    for c in df.columns:
        if any(w in _norm(c) for w in ["fecha","emision","ingreso","salida","entrega","pago","documento"]):
            return c
    return None

# -------------------- parsing & filters --------------------
def parse_time_filters(q:str, hoja:str, df:pd.DataFrame, dic_map:dict) -> List[Dict[str,str]]:
    qn=_norm(q); out=[]
    date_col = _pick_date(df, dic_map, hoja)
    if not date_col: return out
    m = re.search(r"\b(20\d{2})\b", qn)
    if m:
        y=int(m.group(1))
        out.append({"col":date_col,"op":"gte","val":f"{y}-01-01"})
        out.append({"col":date_col,"op":"lte","val":f"{y}-12-31"})
    for mes,num in MESES.items():
        if mes in qn:
            ym = re.search(rf"{mes}\s+de?\s*(20\d{{2}})?", qn)
            y = int(ym.group(1)) if (ym and ym.group(1)) else dt.date.today().year
            d1=dt.date(y,num,1); d2=dt.date(y,12,31) if num==12 else (dt.date(y,num+1,1)-dt.timedelta(days=1))
            out.append({"col":date_col,"op":"gte","val":str(d1)})
            out.append({"col":date_col,"op":"lte","val":str(d2)})
            break
    m2 = re.search(r"últim[oa]s?\s+(\d{1,2})\s+mes", qn)
    if m2:
        n=int(m2.group(1)); today=dt.date.today()
        month_back = today.month-n; year=today.year
        while month_back<=0: month_back+=12; year-=1
        d1=dt.date(year,month_back,1)
        out.append({"col":date_col,"op":"gte","val":str(d1)})
    return out

def guess_plan(q:str, data:Dict[str,pd.DataFrame], dic_map:dict) -> Plan:
    qn=_norm(q)
    if any(w in qn for w in ["sin factur","no facturad","pendientes de factur","por factur"]):
        return Plan(metric="ot_sin_facturar", dims=["CLIENTE"])
    if any(w in qn for w in ["no pagad","pendiente de pago","por cobrar"]):
        return Plan(metric="facturas_no_pagadas", dims=["CLIENTE"])
    if any(w in qn for w in ["aging","vencid","por vencer","cartera"]):
        return Plan(metric="cxc_aging", dims=[])
    if any(w in qn for w in ["lead time","tiempo promedio","taller demora"]):
        return Plan(metric="lead_time_prom", dims=["CLIENTE"])
    if any(w in qn for w in ["margen"]):
        return Plan(metric="margen_pct", dims=["CLIENTE"])
    if any(w in qn for w in ["utilidad","ganancia"]):
        return Plan(metric="utilidad", dims=["CLIENTE"])
    if any(w in qn for w in ["ingreso","venta","facturaci","monto neto","total neto","total ventas"]):
        return Plan(metric="ventas_netas", dims=["CLIENTE"])
    if any(w in qn for w in ["cuantas ot","cuántas ot","numero de ot","número de ot","count ot","cantidad de ot","ot por"]):
        return Plan(metric="ot", op="count", dims=["CLIENTE"])
    return Plan(metric="ventas_netas", dims=["CLIENTE"])

# -------------------- helpers to choose value/dims safely --------------------
POSSIBLE_VALUE_NAMES = [
    "MONTO NETO","MONTO","TOTAL NETO","VALOR NETO","TOTAL","NETO","VALOR","IMPORTE",
]

def _numeric_cols(df: pd.DataFrame) -> List[str]:
    num = df.select_dtypes(include=["number"]).columns.tolist()
    # include object columns that look numeric
    for c in df.columns:
        if c not in num:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                num.append(c)
    return num

def pick_value_col(df: pd.DataFrame, dic_map: dict, hoja: str, plan: Plan, meta_valor: Optional[str]) -> str:
    # 1) metadata "valor" from metrics
    if meta_valor:
        c = resolve_col(df, dic_map, hoja, meta_valor)
        if c and c in df.columns: return c
    # 2) common names
    for name in POSSIBLE_VALUE_NAMES:
        c = resolve_col(df, dic_map, hoja, name, "money") or resolve_col(df, dic_map, hoja, name)
        if c and c in df.columns: return c
    # 3) by role "money" from dictionary
    info = (dic_map.get("map") or {}).get(hoja) or {}
    roles = info.get("roles", {})
    for c, r in roles.items():
        if r=="money" and c in df.columns: return c
    # 4) any numeric column
    nums = _numeric_cols(df)
    if nums: return nums[0]
    # 5) fallback: create a constant column to allow count/sum
    if "_UNO_" not in df.columns:
        df["_UNO_"] = 1
    return "_UNO_"

# -------------------- custom metrics (pandas only; no AI) --------------------
def metric_ot_sin_facturar(data, dic_map) -> pd.DataFrame:
    dfm = data.get("MODELO_BOT", pd.DataFrame()).copy()
    dff = data.get("FINANZAS", pd.DataFrame()).copy()
    if dfm.empty: return pd.DataFrame()
    key = resolve_col(dfm, dic_map, "MODELO_BOT", "OT", "id") or resolve_col(dfm, dic_map, "MODELO_BOT", "PATENTE", "id")
    fecha_entrega = resolve_col(dfm, dic_map, "MODELO_BOT", "FECHA SALIDA PLANTA", "date") or resolve_col(dfm, dic_map, "MODELO_BOT", "FECHA ENTREGA", "date")
    if not key or not fecha_entrega: return pd.DataFrame()
    df_ent = dfm[pd.to_datetime(dfm[fecha_entrega], errors="coerce").notna()].copy()
    if not dff.empty:
        key_fin = resolve_col(dff, dic_map, "FINANZAS", key) or resolve_col(dff, dic_map, "FINANZAS", "OT", "id") or resolve_col(dff, dic_map, "FINANZAS", "PATENTE", "id")
        if key_fin and key_fin in dff.columns:
            billed = set(dff[key_fin].dropna().astype(str).str.strip().unique().tolist())
            df_ent = df_ent[~df_ent[key].astype(str).str.strip().isin(billed)]
    cols = [c for c in [key, fecha_entrega, resolve_col(df_ent, dic_map, "MODELO_BOT","PATENTE"),
                        resolve_col(df_ent, dic_map, "MODELO_BOT","NOMBRE CLIENTE") or resolve_col(df_ent, dic_map, "MODELO_BOT","CLIENTE"),
                        resolve_col(df_ent, dic_map, "MODELO_BOT","MARCA"),
                        resolve_col(df_ent, dic_map, "MODELO_BOT","MODELO")] if c and c in df_ent.columns]
    if not cols:
        return pd.DataFrame()
    out = df_ent[cols].copy()
    out.insert(0, "CATEGORIA", out.get("NOMBRE CLIENTE") if "NOMBRE CLIENTE" in out.columns else out[cols[0]].astype(str))
    out["VALOR"] = 1
    return out[["CATEGORIA","VALOR"]]

def metric_facturas_no_pagadas(data, dic_map) -> pd.DataFrame:
    dff = data.get("FINANZAS", pd.DataFrame()).copy()
    if dff.empty: return pd.DataFrame()
    num = resolve_col(dff, dic_map, "FINANZAS", "NUMERO DE FACTURA", "id") or resolve_col(dff, dic_map, "FINANZAS", "FACTURA")
    estado = resolve_col(dff, dic_map, "FINANZAS", "ESTADO PAGO") or resolve_col(dff, dic_map, "FINANZAS", "ESTADO")
    monto = resolve_col(dff, dic_map, "FINANZAS", "MONTO NETO", "money")
    if estado:
        dff = dff[~dff[estado].astype(str).str.contains("pagad", case=False, na=False)]
    else:
        fecha_pago = resolve_col(dff, dic_map, "FINANZAS", "FECHA PAGO", "date")
        if fecha_pago:
            s = pd.to_datetime(dff[fecha_pago], errors="coerce"); dff = dff[s.isna()]
    if not monto:
        # fall back to any numeric column
        monto = pick_value_col(dff, dic_map, "FINANZAS", Plan("ventas_netas"), meta_valor=None)
    cliente = resolve_col(dff, dic_map, "FINANZAS", "CLIENTE") or resolve_col(dff, dic_map, "FINANZAS", "NOMBRE CLIENTE")
    if cliente and monto in dff.columns:
        res = dff.groupby(cliente)[monto].sum(numeric_only=True).reset_index()
        res.rename(columns={cliente:"CATEGORIA", monto:"VALOR"}, inplace=True)
        return res
    # fallback simple
    return pd.DataFrame({"CATEGORIA":["PENDIENTES"], "VALOR":[float(pd.to_numeric(dff[monto], errors="coerce").sum())]}) if monto in dff.columns else pd.DataFrame()

def metric_cxc_aging(data, dic_map) -> pd.DataFrame:
    dff = data.get("FINANZAS", pd.DataFrame()).copy()
    if dff.empty: return pd.DataFrame()
    fecha = resolve_col(dff, dic_map, "FINANZAS", "FECHA", "date") or resolve_col(dff, dic_map, "FINANZAS", "FECHA EMISION", "date")
    fecha_pago = resolve_col(dff, dic_map, "FINANZAS", "FECHA PAGO", "date")
    monto = resolve_col(dff, dic_map, "FINANZAS", "MONTO NETO", "money") or pick_value_col(dff, dic_map, "FINANZAS", Plan("ventas_netas"), None)
    if not fecha or not monto: return pd.DataFrame()
    dff["_fecha"] = pd.to_datetime(dff[fecha], errors="coerce")
    dff["_fpago"] = pd.to_datetime(dff[fecha_pago], errors="coerce") if fecha_pago else pd.NaT
    today = pd.Timestamp.today().normalize()
    cond_pend = dff["_fpago"].isna() if " _fpago" in dff.columns else dff["_fecha"].notna()
    pend = dff[cond_pend].copy()
    pend["_dias"] = (today - pend["_fecha"]).dt.days
    bins = [-1,30,60,90,99999]
    labels = ["0-30","31-60","61-90","90+"]
    pend["_bucket"] = pd.cut(pend["_dias"], bins=bins, labels=labels)
    res = pend.groupby("_bucket")[monto].sum(numeric_only=True).reset_index()
    res.rename(columns={"_bucket":"CATEGORIA", monto:"VALOR"}, inplace=True)
    return res[["CATEGORIA","VALOR"]]

def metric_lead_time_prom(data, dic_map) -> pd.DataFrame:
    dfm = data.get("MODELO_BOT", pd.DataFrame()).copy()
    if dfm.empty: return pd.DataFrame()
    fin = resolve_col(dfm, dic_map, "MODELO_BOT", "FECHA SALIDA PLANTA", "date") or resolve_col(dfm, dic_map, "MODELO_BOT", "FECHA ENTREGA", "date")
    ini = resolve_col(dfm, dic_map, "MODELO_BOT", "FECHA INGRESO", "date") or resolve_col(dfm, dic_map, "MODELO_BOT", "FECHA RECEPCION", "date")
    cliente = resolve_col(dfm, dic_map, "MODELO_BOT", "NOMBRE CLIENTE") or resolve_col(dfm, dic_map, "MODELO_BOT", "CLIENTE")
    if not fin or not ini: return pd.DataFrame()
    dfm["_ini"] = pd.to_datetime(dfm[ini], errors="coerce")
    dfm["_fin"] = pd.to_datetime(dfm[fin], errors="coerce")
    valid = dfm[dfm["_fin"].notna() & dfm["_ini"].notna()].copy()
    valid["_dias"] = (valid["_fin"] - valid["_ini"]).dt.days
    if cliente:
        res = valid.groupby(cliente)["_dias"].mean().reset_index()
        res.rename(columns={cliente:"CATEGORIA","_dias":"VALOR"}, inplace=True)
        return res
    return pd.DataFrame({"CATEGORIA":["PROMEDIO"], "VALOR":[valid["_dias"].mean()]})

# -------------------- compute --------------------
def compute(plan:Plan, data:Dict[str,pd.DataFrame], dic_map:dict, metrics_cfg:dict) -> pd.DataFrame:
    m = plan.metric or ""
    # custom metrics first
    if m=="ot_sin_facturar": return metric_ot_sin_facturar(data, dic_map)
    if m=="facturas_no_pagadas": return metric_facturas_no_pagadas(data, dic_map)
    if m=="cxc_aging": return metric_cxc_aging(data, dic_map)
    if m=="lead_time_prom": return metric_lead_time_prom(data, dic_map)

    meta = (metrics_cfg.get("metricas") or {}).get(m, {})
    hoja = meta.get("hoja","FINANZAS" if m!="ot" else "MODELO_BOT")
    if hoja not in data: return pd.DataFrame()
    df = data[hoja].copy()
    if df.empty: return pd.DataFrame()

    # choose value column robustly
    meta_valor = meta.get("valor")
    valor = pick_value_col(df, dic_map, hoja, plan, meta_valor)

    # time filters
    plan.filters = (plan.filters or []) + parse_time_filters(" ".join([plan.metric]+plan.dims), hoja, df, dic_map)
    df = _apply_filters(df, plan.filters)

    # dims
    dims_cols=[]
    for d in plan.dims or []:
        col = resolve_col(df, dic_map, hoja, d) or next((c for c in df.columns if _norm(d) in _norm(c)), None)
        if col and col in df.columns: dims_cols.append(col)

    # group or total
    if dims_cols:
        g=df.groupby(dims_cols, dropna=False)
        op = (plan.op or meta.get("op","sum")).lower()
        if op=="count":
            res=g[valor].count().reset_index(name="VALOR") if valor in df.columns else g.size().reset_index(name="VALOR")
        elif op=="avg":
            res=g[valor].mean(numeric_only=True).reset_index(name="VALOR")
        elif op=="max":
            res=g[valor].max(numeric_only=True).reset_index(name="VALOR")
        elif op=="min":
            res=g[valor].min(numeric_only=True).reset_index(name="VALOR")
        else:
            res=g[valor].sum(numeric_only=True).reset_index(name="VALOR")
        res.rename(columns={dims_cols[0]:"CATEGORIA"}, inplace=True)
        return res[["CATEGORIA","VALOR"]] if "VALOR" in res.columns else res
    else:
        op = (plan.op or meta.get("op","sum")).lower()
        if op=="count":
            v = int(df[valor].count()) if valor in df.columns else int(len(df))
            return pd.DataFrame({"CATEGORIA":["TOTAL"],"VALOR":[v]})
        elif op=="avg":
            s = pd.to_numeric(df[valor], errors='coerce') if valor in df.columns else pd.Series([], dtype="float")
            return pd.DataFrame({"CATEGORIA":["TOTAL"],"VALOR":[float(s.mean(skipna=True)) if not s.empty else 0.0]})
        elif op=="max":
            s = pd.to_numeric(df[valor], errors='coerce') if valor in df.columns else pd.Series([], dtype="float")
            return pd.DataFrame({"CATEGORIA":["TOTAL"],"VALOR":[float(s.max(skipna=True)) if not s.empty else 0.0]})
        elif op=="min":
            s = pd.to_numeric(df[valor], errors='coerce') if valor in df.columns else pd.Series([], dtype="float")
            val = float(s.min(skipna=True)) if not s.empty else 0.0
            return pd.DataFrame({"CATEGORIA":["TOTAL"],"VALOR":[val]})
        else:
            s = pd.to_numeric(df[valor], errors='coerce') if valor in df.columns else pd.Series([], dtype="float")
            return pd.DataFrame({"CATEGORIA":["TOTAL"],"VALOR":[float(s.sum(skipna=True)) if not s.empty else 0.0]})

# -------------------- public entry --------------------
def responder(pregunta:str, data:Dict[str,pd.DataFrame], dict_df:Optional[pd.DataFrame], metrics_cfg:dict|None, dicc_static:list|None)->dict:
    metrics_cfg = metrics_cfg or {}
    dic_map = build_dictionary(dict_df, dicc_static)
    plan = guess_plan(pregunta, data, dic_map)
    df = compute(plan, data, dic_map, metrics_cfg)
    return {"plan": plan.__dict__, "df": df}
