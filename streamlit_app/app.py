import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os

# For demonstration purposes, let's define a function that reads & concatenates data
@st.cache_data
def load_data(data_dir="../data"):
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    profiles = []
    esg_scores = []
    executives_list = []
    
    # We'll keep a dictionary {symbol: raw_json} so we can display the raw JSON later
    raw_data_dict = {}

    for file_name in json_files:
        full_path = os.path.join(data_dir, file_name)
        with open(full_path, 'r') as f:
            data = json.load(f)

            # Extract 'profile'
            profile_data = data.get('profile', {})
            symbol = profile_data.get('symbol')
            if symbol:
                raw_data_dict[symbol] = data  # Store raw JSON by symbol

            profiles.append(profile_data)
            
            # Extract 'esg'
            esg_data = data.get('esg', {})
            esg_scores.append(esg_data)
            
            # Extract 'executives'
            execs = data.get('executives', [])
            for e in execs:
                e['symbol'] = profile_data.get('symbol')
            executives_list.extend(execs)
    
    df_profiles = pd.DataFrame(profiles)
    df_esg = pd.DataFrame(esg_scores)
    df_executives = pd.DataFrame(executives_list)

    # Basic cleaning
    df_profiles['companyName'] = df_profiles['companyName'].fillna('Unknown')
    df_esg['ESGScore'] = pd.to_numeric(df_esg['ESGScore'], errors='coerce')

    return df_profiles, df_esg, df_executives, raw_data_dict


def main():
    st.set_page_config(page_title="Company & ESG Dashboard", layout="wide")
    st.title("Company & ESG Dashboard")

    st.write("""
    **This application showcases data from multiple JSON files** (company profiles, executives, and ESG scores)
    in a multi-tab, interactive format. Use the sidebar to select different symbols. Explore each tab for 
    detailed insights on **Company Profile**, **ESG**, **Executives**, and even the **Raw JSON** itself. 
    """)

    # Load data
    df_profiles, df_esg, df_executives, raw_data_dict = load_data()

    # Sidebar filters
    st.sidebar.header("Filters")
    symbols_available = sorted(df_profiles['symbol'].dropna().unique())
    selected_symbol = st.sidebar.selectbox("Select a Symbol", symbols_available)

    # Filter data based on selection
    mask_profile = df_profiles['symbol'] == selected_symbol
    mask_esg = df_esg['symbol'] == selected_symbol
    mask_exec = df_executives['symbol'] == selected_symbol

    selected_profile = df_profiles[mask_profile].iloc[0] if not df_profiles[mask_profile].empty else None
    selected_esg = df_esg[mask_esg].iloc[0] if not df_esg[mask_esg].empty else None
    selected_executives = df_executives[mask_exec]
    selected_raw_json = raw_data_dict.get(selected_symbol, {})

    # Create Streamlit tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "ESG Analysis", "Executives", "Raw JSON"])

    # ------------------- TAB 1: Overview -------------------
    with tab1:
        st.subheader("Company Overview")
        if selected_profile is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Price (USD)",
                          f"${selected_profile['price']:.2f}" if pd.notna(selected_profile['price']) else "N/A")
                if pd.notna(selected_profile['mktCap']):
                    st.metric("Market Cap", f"${selected_profile['mktCap']/1e9:.2f}B")
                else:
                    st.metric("Market Cap", "N/A")
                st.metric("Sector", selected_profile['sector'] if pd.notna(selected_profile['sector']) else "N/A")
                st.metric("Country", selected_profile['country'] if pd.notna(selected_profile['country']) else "N/A")
            with col2:
                st.metric("Beta",
                          str(selected_profile['beta']) if pd.notna(selected_profile['beta']) else "N/A")
                st.metric("Change",
                          f"{selected_profile['changes']}" if pd.notna(selected_profile['changes']) else "N/A")
                st.metric("Range",
                          selected_profile['range'] if pd.notna(selected_profile['range']) else "N/A")
                st.metric("Employees",
                          selected_profile['fullTimeEmployees'] if pd.notna(selected_profile['fullTimeEmployees']) else "N/A")

            st.write("**Company Description:**")
            st.write(selected_profile['description'])

            # Extra info in an expander
            with st.expander("More Profile Details"):
                st.write(f"**CEO**: {selected_profile.get('ceo', 'N/A')}")
                st.write(f"**Phone**: {selected_profile.get('phone', 'N/A')}")
                st.write(f"**Address**: {selected_profile.get('address', 'N/A')}, "
                         f"{selected_profile.get('city', '')}, {selected_profile.get('state', '')} {selected_profile.get('zip', '')}")
                st.write(f"**Exchange**: {selected_profile.get('exchange', 'N/A')} ({selected_profile.get('exchangeShortName', 'N/A')})")
                st.write(f"**Website**: {selected_profile.get('website', 'N/A')}")
                st.write(f"**Industry**: {selected_profile.get('industry', 'N/A')}")
                st.write(f"**Sector**: {selected_profile.get('sector', 'N/A')}")

        else:
            st.warning("No profile data available for this symbol.")

    # ------------------- TAB 2: ESG Analysis -------------------
    with tab2:
        st.subheader("ESG Analysis")
        if selected_esg is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Environmental", selected_esg.get('environmentalScore', 'N/A'))
            col2.metric("Social", selected_esg.get('socialScore', 'N/A'))
            col3.metric("Governance", selected_esg.get('governanceScore', 'N/A'))
            col4.metric("Overall ESG", selected_esg.get('ESGScore', 'N/A'))

            # We can also create a small bar chart comparing E, S, G
            esg_scores = [
                {"Category": "Environmental", "Score": selected_esg.get('environmentalScore', None)},
                {"Category": "Social", "Score": selected_esg.get('socialScore', None)},
                {"Category": "Governance", "Score": selected_esg.get('governanceScore', None)},
            ]
            df_esg_scores = pd.DataFrame(esg_scores).dropna()  # drop rows with None

            if not df_esg_scores.empty:
                fig_esg_bar = px.bar(
                    df_esg_scores,
                    x="Category",
                    y="Score",
                    color="Category",
                    title="ESG Score Comparison",
                    range_y=[0, 100],  # assuming ESG is on a 0-100 scale
                )
                fig_esg_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_esg_bar, use_container_width=True)

            # Additional ESG details
            with st.expander("More ESG Details"):
                st.write(f"**Industry**: {selected_esg.get('industry', 'N/A')}")
                st.write(f"**Form Type**: {selected_esg.get('formType', 'N/A')}")
                st.write(f"**Date**: {selected_esg.get('date', 'N/A')}")
                st.write(f"**SEC Filing**: {selected_esg.get('url', 'N/A')}")
        else:
            st.warning("No ESG data available for this symbol.")

    # ------------------- TAB 3: Executives -------------------
    with tab3:
        st.subheader("Executives & Pay")
        if not selected_executives.empty:
            # 1. Simple bar chart for executive pay (where not null)
            execs_with_pay = selected_executives.dropna(subset=['pay'])
            if not execs_with_pay.empty:
                fig_pay = px.bar(
                    execs_with_pay,
                    x='name',
                    y='pay',
                    color='name',
                    labels={'pay': 'Pay (USD)', 'name': 'Executive'},
                    title='Executive Pay Breakdown'
                )
                st.plotly_chart(fig_pay, use_container_width=True)
            else:
                st.info("No pay information available for these executives.")

            # 2. Tabular display of all executives
            st.dataframe(
                selected_executives[['title', 'name', 'pay', 'currencyPay', 'gender', 'yearBorn']].reset_index(drop=True)
            )
        else:
            st.warning("No executive data available for this symbol.")

    # ------------------- TAB 4: Raw JSON -------------------
    with tab4:
        st.subheader("Raw JSON Data")
        if selected_raw_json:
            st.write("Below is the raw JSON data for this symbol. "
                     "It can be useful for debugging or deeper exploration of fields.")
            st.json(selected_raw_json)
        else:
            st.warning("No raw JSON found for this symbol.")


if __name__ == "__main__":
    main()
