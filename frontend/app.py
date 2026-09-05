from datetime import date
import requests
import streamlit as st

st.set_page_config(
    page_title="Support Ticket Desk", page_icon="🎫", layout="wide"
)

API_ENDPOINT = "http://127.0.0.1:8000/predict"

st.title("🎫 Customer Support Portal & Ticket Triage")
st.caption(
    "Automated ticket dispatch and category prediction powered by Scikit-Learn"
    " & FastAPI."
)

with st.form("ticket_submission_form"):
  st.subheader("1. Customer Profile")
  col1, col2 = st.columns(2)

  with col1:
    cust_name = st.text_input("Customer Name", placeholder="e.g. Jane Doe")
    cust_email = st.text_input(
        "Customer Email", placeholder="jane.doe@example.com"
    )
    cust_age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)

  with col2:
    cust_gender = st.selectbox("Customer Gender", ["Female", "Male", "Other"])
    product_purchased = st.selectbox(
        "Product Purchased",
        [
            "GoPro Hero",
            "LG Smart TV",
            "Dell XPS",
            "Microsoft Office",
            "Autodesk AutoCAD",
            "Nintendo Switch",
            "iPhone 14",
            "PlayStation 5",
        ],
    )
    date_of_purchase = st.date_input("Date of Purchase", value=date.today())

  st.divider()
  st.subheader("2. Issue Details")

  ticket_channel = st.selectbox("Channel", ["Email", "Chat", "Social media", "Phone"])
  ticket_priority = st.select_slider(
      "Priority",
      options=["Low", "Medium", "Critical"],
      value="Medium",
  )
  ticket_subject = st.text_input(
      "Ticket Subject",
      placeholder="e.g. Account access lock / Hardware overheating",
  )
  ticket_description = st.text_area(
      "Ticket Description",
      placeholder="Explain the detailed issue you are facing...",
      height=150,
  )

  submitted = st.form_submit_button(
      "Submit Ticket", use_container_width=True, type="primary"
  )

if submitted:
  if not cust_name or not cust_email or not ticket_subject or not ticket_description:
    st.error("Please fill in all mandatory text fields before submitting.")
  else:
    payload = {
        "customer_name": cust_name,
        "customer_email": cust_email,
        "customer_age": int(cust_age),
        "customer_gender": cust_gender,
        "product_purchased": product_purchased,
        "ticket_subject": ticket_subject,
        "ticket_description": ticket_description,
        "ticket_priority": ticket_priority,
        "ticket_channel": ticket_channel,
    }

    with st.spinner("Processing ticket and routing via ML model..."):
      try:
        response = requests.post(API_ENDPOINT, json=payload, timeout=10)
        if response.status_code == 200:
          data = response.json()
          st.success("Ticket Successfully Classified and Dispatched!")

          res_col1, res_col2, res_col3 = st.columns(3)
          res_col1.metric("Predicted Ticket Type", data["ticket_type"])
          res_col2.metric("Assigned Queue", data["assigned_queue"])
          res_col3.metric(
              "Confidence Score",
              f"{data['confidence_score']:.2%}"
              if data["confidence_score"]
              else "N/A",
          )
        else:
          st.error(f"Server Error ({response.status_code}): {response.text}")
      except requests.exceptions.ConnectionError:
        st.error(
            "Could not reach backend API. Ensure FastAPI server is running on"
            " port 8000."
        )