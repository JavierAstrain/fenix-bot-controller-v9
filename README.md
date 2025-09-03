
# Fénix Controller — Paquete final (Google Sheets fijo + motor determinístico)

## 1) Instalar dependencias
```bash
pip install -r requirements.txt
```

## 2) Configurar `.streamlit/secrets.toml`
```toml
USER = "admin"
PASSWORD = "1234"
GSHEET_URL = "https://docs.google.com/spreadsheets/d/1SaXuzhY_sJ9Tk9MOLDLAI4OVdsNbCP-X4L8cP15yTqo/edit?gid=758257628#gid=758257628"

# Cualquiera de estas claves sirve (usa una):
GOOGLE_CREDENTIALS = """{ ...JSON service account... }"""
# GOOGLE_SERVICE_ACCOUNT_JSON = """{ ... }"""
# GOOGLE_SERVICE_ACCOUNT = """{ ... }"""
# SERVICE_ACCOUNT_JSON = """{ ... }"""
# GCP_SERVICE_ACCOUNT_JSON = """{ ... }"""

# opcional para narrativa IA:
# OPENAI_API_KEY = "sk-..."
```

## 3) Ejecutar
```bash
streamlit run app.py
```

## Qué hay
- `app.py` — Menú completo, hoja fija en Drive (sin uploader), narrativa izquierda + tabla/gráfico derecha.
- `engine_semantico.py` — motor determinístico (ventas, OT, OT sin facturar, facturas no pagadas, aging CxC, lead time prom).
- `diccionario.yaml` + `diccionario_static.json` — diccionario MEGA robusto generado a partir de tu Excel + alias canónicos.
- `metrics.yaml` — catálogo de métricas.
- Logos: `Nexa_logo.png`, `Isotipo_Nexa.png`, `Fenix_isotipo.png`.
