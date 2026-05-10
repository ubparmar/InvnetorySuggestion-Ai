import io
import os
import math
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None


APP_TITLE = "AI Vape Inventory Order Assistant"
DEFAULT_CSV_PATH = Path("vape_inventory_dummy_v1.csv")

# LOCAL TESTING ONLY:
# Paste your NEW Gemini API key below. Do not use the exposed key from your screenshot.
# For production, use .streamlit/secrets.toml or environment variables instead.
GEMINI_API_KEY = "AIzaSyDFt6LL7Stm8tCotoVoB98snEiI5SF-H5c"

REQUIRED_COLUMNS = [
    "sku",
    "brand",
    "product_name",
    "supplier",
    "previous_inventory_qty",
    "current_stock",
    "past_month_units_sold",
    "past_month_purchase_qty",
    "reorder_point",
    "lead_time_days",
    "safety_stock_days",
    "pack_size",
    "min_order_qty",
    "unit_cost",
]

NUMERIC_COLUMNS = [
    "previous_inventory_qty",
    "current_stock",
    "past_month_units_sold",
    "past_month_purchase_qty",
    "reorder_point",
    "lead_time_days",
    "safety_stock_days",
    "pack_size",
    "min_order_qty",
    "unit_cost",
]


def get_api_key() -> str:
    """Read API key from direct variable first, then Streamlit secrets, then environment variable."""
    direct_key = GEMINI_API_KEY.strip()
    if direct_key and direct_key != "PASTE_YOUR_NEW_GEMINI_API_KEY_HERE":
        return direct_key

    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""

    return key or os.environ.get("GEMINI_API_KEY", "")

def load_inventory(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    if DEFAULT_CSV_PATH.exists():
        return pd.read_csv(DEFAULT_CSV_PATH)

    return pd.DataFrame()


def validate_inventory(df: pd.DataFrame):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return False, missing
    return True, []


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def round_up_to_pack_and_min_order(qty: float, pack_size: float, min_order_qty: float) -> int:
    qty = max(float(qty), 0)
    pack_size = max(int(pack_size), 1)
    min_order_qty = max(int(min_order_qty), 0)

    if qty <= 0:
        return 0

    qty = max(qty, min_order_qty)
    return int(math.ceil(qty / pack_size) * pack_size)


def calculate_order_suggestions(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_numeric_columns(df)

    df["avg_daily_sales"] = df["past_month_units_sold"] / 30
    df["lead_time_demand"] = df["avg_daily_sales"] * df["lead_time_days"]
    df["safety_stock_qty"] = df["avg_daily_sales"] * df["safety_stock_days"]
    df["target_stock"] = (df["lead_time_demand"] + df["safety_stock_qty"]).apply(math.ceil)

    df["target_gap"] = (df["target_stock"] - df["current_stock"]).clip(lower=0)
    df["reorder_point_gap"] = (df["reorder_point"] - df["current_stock"]).clip(lower=0)
    df["raw_suggested_order_qty"] = df[["target_gap", "reorder_point_gap"]].max(axis=1)

    df["suggested_order_qty"] = df.apply(
        lambda row: round_up_to_pack_and_min_order(
            row["raw_suggested_order_qty"],
            row["pack_size"],
            row["min_order_qty"],
        ),
        axis=1,
    )

    df["estimated_order_cost"] = df["suggested_order_qty"] * df["unit_cost"]

    def priority(row):
        if row["suggested_order_qty"] <= 0:
            return "No Order"
        if row["current_stock"] <= row["reorder_point"] * 0.5:
            return "Urgent"
        if row["current_stock"] <= row["reorder_point"]:
            return "High"
        return "Medium"

    def reason(row):
        if row["suggested_order_qty"] <= 0:
            return "Enough stock based on current demand."
        if row["current_stock"] <= row["reorder_point"]:
            return "Stock is at or below reorder point."
        return "Projected demand during lead time and safety period is higher than current stock."

    df["priority"] = df.apply(priority, axis=1)
    df["order_reason"] = df.apply(reason, axis=1)

    return df


def build_ai_prompt(order_df: pd.DataFrame, store_context: str) -> str:
    order_items = order_df[order_df["suggested_order_qty"] > 0].copy()

    if order_items.empty:
        return "No inventory order is required. Explain this in one short paragraph."

    total_cost = float(order_items["estimated_order_cost"].sum())
    payload = order_items[
        [
            "sku",
            "brand",
            "product_name",
            "supplier",
            "current_stock",
            "past_month_units_sold",
            "avg_daily_sales",
            "reorder_point",
            "lead_time_days",
            "safety_stock_days",
            "suggested_order_qty",
            "estimated_order_cost",
            "priority",
            "order_reason",
        ]
    ].to_dict(orient="records")

    return f"""
You are an inventory planning assistant for a regulated retail inventory system.

Store context:
{store_context or "No extra context provided."}

Task:
Create a concise ordering summary based only on the calculated order data below.

Important rules:
- Do not change SKU names or calculated quantities.
- Do not invent products.
- Group insights by supplier/brand where useful.
- Highlight urgent/high priority items.
- Mention total estimated order cost.
- Keep the tone professional and practical.
- Add a short compliance reminder that regulated products should follow local laws and store policy.

Total estimated order cost: ${total_cost:.2f}

Calculated order data:
{json.dumps(payload, indent=2)}
"""


def generate_ai_summary(prompt: str, model_name: str, enable_google_search: bool, thinking_level: str) -> str:
    api_key = get_api_key()

    if not api_key:
        return "AI summary skipped: GEMINI_API_KEY is not set in .streamlit/secrets.toml or environment variables."

    if genai is None or types is None:
        return "AI summary skipped: google-genai is not installed. Run: pip install google-genai"

    client = genai.Client(api_key=api_key)

    config = None
    config_kwargs = {}

    if enable_google_search:
        try:
            config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
        except Exception:
            try:
                config_kwargs["tools"] = [types.Tool(googleSearch=types.GoogleSearch())]
            except Exception:
                pass

    if thinking_level != "OFF":
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
        except Exception:
            pass

    if config_kwargs:
        try:
            config = types.GenerateContentConfig(**config_kwargs)
        except Exception:
            config = None

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        return response.text or "AI returned an empty response."
    except Exception as first_error:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return response.text or "AI returned an empty response."
        except Exception as second_error:
            return (
                "AI generation failed. Check your API key, model name, quota, and installed google-genai version.\n\n"
                f"Error: {second_error}\n\n"
                f"First attempt error: {first_error}"
            )


def make_pdf(order_df: pd.DataFrame, ai_summary: str) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        rightMargin=24,
        leftMargin=24,
        topMargin=24,
        bottomMargin=24,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Generated Order Recommendation", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    order_items = order_df[order_df["suggested_order_qty"] > 0].copy()
    total_cost = float(order_items["estimated_order_cost"].sum()) if not order_items.empty else 0.0

    story.append(Paragraph("AI Summary", styles["Heading2"]))
    summary_text = (ai_summary or "No AI summary generated.").replace("\n", "<br/>")
    story.append(Paragraph(summary_text, styles["BodyText"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Order Items", styles["Heading2"]))

    if order_items.empty:
        story.append(Paragraph("No items need ordering at this time.", styles["BodyText"]))
    else:
        order_items = order_items.sort_values(["supplier", "priority", "brand", "product_name"])
        table_data = [[
            "SKU",
            "Product",
            "Supplier",
            "Stock",
            "Sold 30D",
            "Reorder Pt",
            "Target",
            "Order Qty",
            "Unit Cost",
            "Est. Cost",
            "Priority",
        ]]

        for _, row in order_items.iterrows():
            table_data.append([
                str(row["sku"]),
                str(row["product_name"]),
                str(row["supplier"]),
                str(int(row["current_stock"])),
                str(int(row["past_month_units_sold"])),
                str(int(row["reorder_point"])),
                str(int(row["target_stock"])),
                str(int(row["suggested_order_qty"])),
                f"${float(row['unit_cost']):.2f}",
                f"${float(row['estimated_order_cost']):.2f}",
                str(row["priority"]),
            ])

        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.black),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (3, 1), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Total Estimated Order Cost: ${total_cost:.2f}", styles["Heading3"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Calculation Method", styles["Heading2"]))
    story.append(Paragraph(
        "Average daily sales = past month units sold / 30. "
        "Target stock = average daily sales × (lead time days + safety stock days). "
        "Suggested order quantity = max(target stock gap, reorder point gap, 0), rounded up by pack size and minimum order quantity.",
        styles["BodyText"],
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


st.set_page_config(page_title=APP_TITLE, page_icon="📦", layout="wide")

st.title("📦 AI Inventory Order Assistant — CSV Version")
st.caption("Version 1: no database, CSV-based inventory + AI order summary + PDF generation")

with st.sidebar:
    st.header("AI Settings")
    model_name = st.text_input("Gemini/Gemma model", value="gemma-4-31b-it")
    thinking_level = st.selectbox("Thinking level", ["OFF", "LOW", "MEDIUM", "HIGH"], index=3)
    enable_google_search = st.toggle("Enable Google Search tool", value=False)

    st.divider()
    st.header("CSV")
    uploaded_file = st.file_uploader("Upload inventory CSV", type=["csv"])
    save_local = st.toggle("Save edited CSV locally", value=False)

    st.divider()
    api_key_found = bool(get_api_key())
    st.write("API key status:", "✅ Found" if api_key_found else "❌ Missing")

inventory_df = load_inventory(uploaded_file)

if inventory_df.empty:
    st.info("Upload your inventory CSV, or place vape_inventory_dummy_v1.csv in the same folder as app.py.")
    st.stop()

is_valid, missing_cols = validate_inventory(inventory_df)
if not is_valid:
    st.error("Your CSV is missing required columns:")
    st.write(missing_cols)
    st.stop()

st.subheader("1. Inventory Data")
st.write("Edit values directly here. The app recalculates suggested order quantities after editing.")

editable_df = st.data_editor(
    inventory_df,
    use_container_width=True,
    num_rows="dynamic",
    key="inventory_editor",
)

calculated_df = calculate_order_suggestions(editable_df)

if save_local:
    try:
        calculated_df.to_csv(DEFAULT_CSV_PATH, index=False)
        st.success(f"Saved locally to {DEFAULT_CSV_PATH}")
    except Exception as e:
        st.warning(f"Could not save file locally: {e}")

st.subheader("2. Order Dashboard")

brand_options = sorted(calculated_df["brand"].dropna().unique().tolist())
priority_options = ["Urgent", "High", "Medium", "No Order"]

filter_col1, filter_col2, filter_col3 = st.columns(3)
with filter_col1:
    selected_brands = st.multiselect("Filter by brand", brand_options, default=brand_options)
with filter_col2:
    selected_priorities = st.multiselect("Filter by priority", priority_options, default=priority_options)
with filter_col3:
    only_order_needed = st.toggle("Show only items needing order", value=False)

view_df = calculated_df.copy()
view_df = view_df[view_df["brand"].isin(selected_brands)]
view_df = view_df[view_df["priority"].isin(selected_priorities)]
if only_order_needed:
    view_df = view_df[view_df["suggested_order_qty"] > 0]

order_needed_df = calculated_df[calculated_df["suggested_order_qty"] > 0]
urgent_df = calculated_df[calculated_df["priority"] == "Urgent"]
total_estimated_cost = float(order_needed_df["estimated_order_cost"].sum()) if not order_needed_df.empty else 0.0

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Total SKUs", len(calculated_df))
metric2.metric("Items to Order", len(order_needed_df))
metric3.metric("Urgent Items", len(urgent_df))
metric4.metric("Estimated Cost", f"${total_estimated_cost:,.2f}")

st.dataframe(
    view_df[
        [
            "sku",
            "brand",
            "product_name",
            "supplier",
            "current_stock",
            "past_month_units_sold",
            "avg_daily_sales",
            "reorder_point",
            "target_stock",
            "suggested_order_qty",
            "estimated_order_cost",
            "priority",
            "order_reason",
        ]
    ],
    use_container_width=True,
)

st.subheader("3. Supplier Order Summary")

if order_needed_df.empty:
    st.success("No order needed based on current CSV data.")
else:
    supplier_summary = (
        order_needed_df.groupby("supplier", as_index=False)
        .agg(
            items_to_order=("sku", "count"),
            total_units=("suggested_order_qty", "sum"),
            estimated_cost=("estimated_order_cost", "sum"),
        )
        .sort_values("estimated_cost", ascending=False)
    )
    st.dataframe(supplier_summary, use_container_width=True)

st.subheader("4. Generate AI Summary + PDF")

store_context = st.text_area(
    "Optional store/order context for AI",
    placeholder="Example: Small convenience store. Prefer conservative orders. Avoid overstocking slow movers.",
)

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.download_button(
        "Download Updated CSV",
        data=to_csv_bytes(calculated_df),
        file_name="updated_inventory_with_order_suggestions.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_b:
    if st.button("Generate AI Summary", use_container_width=True):
        prompt = build_ai_prompt(calculated_df, store_context)
        with st.spinner("Generating AI order summary..."):
            st.session_state["ai_summary"] = generate_ai_summary(
                prompt=prompt,
                model_name=model_name,
                enable_google_search=enable_google_search,
                thinking_level=thinking_level,
            )

with col_c:
    if st.button("Generate PDF", use_container_width=True):
        if "ai_summary" not in st.session_state:
            prompt = build_ai_prompt(calculated_df, store_context)
            with st.spinner("Generating AI order summary first..."):
                st.session_state["ai_summary"] = generate_ai_summary(
                    prompt=prompt,
                    model_name=model_name,
                    enable_google_search=enable_google_search,
                    thinking_level=thinking_level,
                )
        st.session_state["pdf_bytes"] = make_pdf(calculated_df, st.session_state["ai_summary"])

if "ai_summary" in st.session_state:
    st.markdown("### AI Order Summary")
    st.write(st.session_state["ai_summary"])

if "pdf_bytes" in st.session_state:
    st.download_button(
        "Download Generated Order PDF",
        data=st.session_state["pdf_bytes"],
        file_name="generated_inventory_order.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

with st.expander("Required CSV columns"):
    st.code("\n".join(REQUIRED_COLUMNS))

with st.expander("API key setup"):
    st.code(
        """
# Option 1, simplest local testing:
# Paste your NEW key near the top of app.py:
GEMINI_API_KEY = "paste_your_new_key_here"

# Option 2, safer/recommended:
# Create .streamlit/secrets.toml and add:
GEMINI_API_KEY = "paste_your_new_key_here"
        """.strip(),
        language="python",
    )
