
import os, json
from pathlib import Path
from typing import Dict
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

from engine_semantico import responder as sem_responder
from yaml import safe_load

# --- Branding ---
st.set_page_config(layout="wide", page_title="Fénix Controller", page_icon="Fenix_isotipo.png")
if Path("Isotipo_Nexa.png").exists():
    st.sidebar.image("Isotipo_Nexa.png", width=42)
st.sidebar.markdown("### Nexa")
st.sidebar.markdown("---")

# --- Config ---
DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1SaXuzhY_sJ9Tk9MOLDLAI4OVdsNbCP-X4L8cP15yTqo/edit?gid=758257628#gid=758257628"
GSHEET_URL = st.secrets.get("GSHEET_URL", DEFAULT_SHEET_URL)
ALLOWED_SHEETS = {"MODELO_BOT","FINANZAS","DICCIONARIO"}
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", None)

# --- Auth ---
USER = st.secrets.get("USER", "admin")
PASSWORD = st.secrets.get("PASSWORD", "1234")
if "auth" not in st.session_state:
    st.session_state.auth = False
def login():
    u = st.sidebar.text_input("Usuario")
    p = st.sidebar.text_input("Contraseña", type="password")
    if st.sidebar.button("Entrar"):
        if u==USER and p==PASSWORD:
            st.session_state.auth=True; st.rerun()
        else:
            st.sidebar.error("Credenciales inválidas.")
if not st.session_state.auth: login(); st.stop()

menu = st.sidebar.radio("Menú", ["Datos","Vista previa","KPIs","Consulta IA","Historial","Uso de Tokens","Diagnóstico IA","Soporte"])
st.sidebar.markdown("---")
if st.sidebar.button("🔒 Cerrar sesión"):
    st.session_state.auth=False; st.experimental_rerun()

# --- Cred helper robusto ---
def _get_service_account_json():
    candidates = [
        "GOOGLE_CREDENTIALS","GOOGLE_SERVICE_ACCOUNT_JSON","GOOGLE_SERVICE_ACCOUNT","GCP_SERVICE_ACCOUNT_JSON","SERVICE_ACCOUNT_JSON"
    ]
    for key in candidates:
        try:
            v = st.secrets.get(key)
        except Exception:
            v = None
        if v is None:
            v = os.getenv(key)
        if not v: 
            continue
        if isinstance(v, dict):
            if "private_key" in v: v["private_key"] = v["private_key"].replace("\\n","\n")
            return json.dumps(v)
        s = str(v).strip().strip("'").strip('"')
        try:
            d = json.loads(s)
            if "private_key" in d: d["private_key"] = d["private_key"].replace("\\n","\n")
            return json.dumps(d)
        except Exception:
            try:
                s2 = bytes(s, "utf-8").decode("unicode_escape")
                d = json.loads(s2)
                if "private_key" in d: d["private_key"] = d["private_key"].replace("\\n","\n")
                return json.dumps(d)
            except Exception:
                pass
    # pasted
    try:
        pasted = st.session_state.get("pasted_json")
        if pasted:
            d = json.loads(pasted); 
            if "private_key" in d: d["private_key"] = d["private_key"].replace("\\n","\n")
            return json.dumps(d)
    except Exception:
        pass
    return None

# --- Loader Google Sheets ---
@st.cache_data(show_spinner=False, ttl=300)
def load_gsheet(sheet_url: str, raw_json: str) -> Dict[str,pd.DataFrame]:
    if not raw_json: 
        raise RuntimeError("Faltan credenciales de Service Account (GOOGLE_CREDENTIALS ó GOOGLE_SERVICE_ACCOUNT_JSON).")
    info = json.loads(raw_json)
    creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"])
    cli = gspread.authorize(creds)
    ss = cli.open_by_url(sheet_url)
    data = {}
    for ws in ss.worksheets():
        name = ws.title.strip().upper()
        if name in ALLOWED_SHEETS:
            data[name] = pd.DataFrame(ws.get_all_records())
    return data

RAW_CREDS = _get_service_account_json()
try:
    data = load_gsheet(GSHEET_URL, RAW_CREDS) if RAW_CREDS else {}
except Exception as e:
    st.error(f"No pude cargar Google Sheets: {e}")
    st.info("Claves disponibles en secrets: " + str(list(st.secrets.keys())))
    data = {}

dict_df = data.get("DICCIONARIO")
# Diccionario estático de respaldo
try:
    dicc_static = json.loads(Path("diccionario_static.json").read_text(encoding="utf-8"))
except Exception:
    try:
        dicc_static = safe_load(Path("diccionario.yaml").read_text(encoding="utf-8"))
    except Exception:
        dicc_static = []

# Métricas
try:
    METRICS = safe_load(Path("metrics.yaml").read_text(encoding="utf-8"))
except Exception:
    METRICS = {"metricas":{}}

# --- Utils de presentación ---
def _normalize_df_result(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    out = df.copy()
    if out.columns[0] != "CATEGORIA":
        out.rename(columns={out.columns[0]:"CATEGORIA"}, inplace=True)
    if len(out.columns)>=2 and out.columns[1] != "VALOR":
        out.rename(columns={out.columns[1]:"VALOR"}, inplace=True)
    return out

def _narrate(question: str, df: pd.DataFrame) -> str:
    if not OPENAI_KEY:
        if df is None or df.empty: return "No encontré registros que cumplan el criterio."
        total = float(pd.to_numeric(df["VALOR"], errors="coerce").sum()) if "VALOR" in df.columns else 0.0
        top = df.sort_values("VALOR", ascending=False).head(3) if "VALOR" in df.columns else df.head(3)
        lines = [f"**Respuesta**: cálculo determinístico para _{question}_.",
                 f"**Total**: {int(total):,}".replace(",", ".")]
        if not top.empty:
            tops = ", ".join([f"{r['CATEGORIA']} ({int(r['VALOR']):,})".replace(",", ".") for _, r in top.iterrows()])
            lines.append(f"**Top categorías**: {tops}")
        lines.append("**Proyecciones**: base, +10%, -10% sobre el total.")
        lines.append("**Recomendaciones**: atacar top categorías, revisar outliers, auditar OT sin facturar / facturas no pagadas.")
        return "\n\n".join(lines)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        sample = df.head(50).to_dict(orient="records") if df is not None else []
        prompt = f"""Eres controller financiero/operaciones. Responde en español, claro y ejecutivo:
- Respuesta concreta a la pregunta
- Diagnóstico breve
- Proyecciones (+10%, -10%)
- 3 recomendaciones accionables
Pregunta: {question}
Tabla (primeras filas): {sample}"""
        chat = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.2)
        return chat.choices[0].message.content
    except Exception as ex:
        return f"(Narrador IA desactivado: {ex})"

# --- Vistas ---
def view_datos():
    st.title("Datos")
    if not data: st.warning("Sin datos cargados."); return
    sel = st.selectbox("Hoja", sorted(list(data.keys())))
    st.dataframe(data[sel], use_container_width=True)

def view_preview():
    st.title("Vista previa / Esquema")
    if not data: st.warning("Sin datos cargados."); return
    for hoja, df in data.items():
        st.subheader(hoja)
        st.dataframe(df.head(10), use_container_width=True)

def view_kpis():
    st.title("KPIs")
    if not data: st.warning("Sin datos cargados."); return
    # placeholder de KPIs: mantener integración si existe analizador
    try:
        from analizador import analizar_datos_taller
        kpis = analizar_datos_taller(data, cliente_contiene="")
    except Exception as e:
        kpis = {"detalle":"analizador no disponible", "error":str(e)}
    st.json(kpis)

def view_consulta():
    st.title("Consulta IA")
    if not data: st.warning("Sin datos cargados."); return
    if sem_responder is None:
        st.error("engine_semantico.py no disponible.")
        return
    q = st.text_input("Pregunta al negocio", placeholder="Ej: Vehículos entregados sin facturar • Monto neto por cliente en agosto 2024 • ¿Cuántas OT por mes este año?")
    if st.button("Responder"):
        res = None
        # firmas posibles
        try:
            res = sem_responder(q, data, dict_df, METRICS, dicc_static)
        except TypeError:
            res = sem_responder(q, data, dict_df, dicc_static)
        df = _normalize_df_result(res.get("df")) if isinstance(res, dict) else None
        plan = res.get("plan", {}) if isinstance(res, dict) else {}
        left, right = st.columns([1.15,1])
        with left:
            st.subheader("Respuesta en lenguaje natural")
            st.markdown(_narrate(q, df))
            st.markdown("**Plan de cálculo**")
            st.json(plan, expanded=False)
        with right:
            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True)
                try:
                    st.bar_chart(df.set_index("CATEGORIA")["VALOR"])
                except Exception:
                    pass
                st.download_button("⬇️ CSV", df.to_csv(index=False).encode("utf-8"), "resultado.csv", "text/csv")
            else:
                st.info("Sin resultados. Ajusta filtros/consulta.")

def view_historial():
    st.title("Historial")
    st.json(st.session_state.get("historial", []))

def view_tokens():
    st.title("Uso de Tokens")
    if not OPENAI_KEY:
        st.warning("OpenAI no configurado.")
    else:
        st.info("Instrumenta _narrate() para registrar usage por respuesta.")

def view_diag():
    st.title("Diagnóstico IA / Conexión")
    st.write("Google Sheet URL:", GSHEET_URL)
    st.write("Claves disponibles en secrets:", list(st.secrets.keys()))
    with st.expander("Pegar credenciales (si no están en secrets)"):
        pasted = st.text_area("JSON del Service Account", height=220, placeholder='{"type":"service_account", ...}')
        if st.button("Usar credenciales pegadas"):
            st.session_state.pasted_json = pasted
            st.success("Guardadas en sesión. Recarga la app.")
    if data:
        for k, df in data.items():
            st.write(k, df.shape, list(df.columns)[:12])

def view_support():
    st.title("Soporte")
    st.write("Escríbenos con capturas y la pregunta exacta a soporte@nexa.cl")

# --- Router ---
if menu=="Datos": view_datos()
elif menu=="Vista previa": view_preview()
elif menu=="KPIs": view_kpis()
elif menu=="Consulta IA": view_consulta()
elif menu=="Historial": view_historial()
elif menu=="Uso de Tokens": view_tokens()
elif menu=="Diagnóstico IA": view_diag()
else: view_support()
