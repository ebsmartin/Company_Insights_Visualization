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

    for file_name in json_files:
        with open(os.path.join(data_dir, file_name), 'r') as f:
            data = json.load(f)

            # Extract 'profile'
            profile_data = data.get('profile', {})
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

    return df_profiles, df_esg, df_executives

def main():
    st.title("Company Profiles & ESG Dashboard")
    st.write("""
        This application showcases data from multiple JSON files (company profiles, executives, and ESG scores) 
        in an interactive format. Select different symbols or metrics for deeper insights.
    """)

    # Load data
    df_profiles, df_esg, df_executives = load_data()

    # Sidebar filters
    st.sidebar.header("Filters")
    symbols_available = sorted(df_profiles['symbol'].dropna().unique())
    selected_symbol = st.sidebar.selectbox("Select a Symbol", symbols_available)

    # Filter data based on selection
    selected_profile = df_profiles[df_profiles['symbol'] == selected_symbol].iloc[0] if not df_profiles[df_profiles['symbol'] == selected_symbol].empty else None
    selected_esg = df_esg[df_esg['symbol'] == selected_symbol].iloc[0] if not df_esg[df_esg['symbol'] == selected_symbol].empty else None
    selected_executives = df_executives[df_executives['symbol'] == selected_symbol]

    # Display profile info
    st.subheader("Company Profile")
    if selected_profile is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Price (USD)", f"${selected_profile['price']:.2f}" if pd.notna(selected_profile['price']) else "N/A")
            st.metric("Market Cap", f"${selected_profile['mktCap']/1e9:.2f}B" if pd.notna(selected_profile['mktCap']) else "N/A")
            st.metric("Sector", selected_profile['sector'] if pd.notna(selected_profile['sector']) else "N/A")
        with col2:
            st.metric("Beta", selected_profile['beta'] if pd.notna(selected_profile['beta']) else "N/A")
            st.metric("Change", f"{selected_profile['changes']}" if pd.notna(selected_profile['changes']) else "N/A")
            st.metric("Range", selected_profile['range'] if pd.notna(selected_profile['range']) else "N/A")
        
        st.write("**Company Description:**")
        st.write(selected_profile['description'])
        
    # Display ESG info
    st.subheader("ESG Information")
    if selected_esg is not None:
        st.write(f"**Environmental Score**: {selected_esg.get('environmentalScore', 'N/A')}")
        st.write(f"**Social Score**: {selected_esg.get('socialScore', 'N/A')}")
        st.write(f"**Governance Score**: {selected_esg.get('governanceScore', 'N/A')}")
        st.write(f"**Overall ESG Score**: {selected_esg.get('ESGScore', 'N/A')}")
    else:
        st.write("No ESG data available for this symbol.")

    # Display Exec Info
    st.subheader("Executives & Pay")
    if not selected_executives.empty:
        fig_pay = px.bar(
            selected_executives.dropna(subset=['pay']),
            x='name', 
            y='pay',
            color='name',
            labels={'pay': 'Pay (USD)', 'name': 'Executive'},
            title='Executive Pay Breakdown'
        )
        st.plotly_chart(fig_pay, use_container_width=True)
        
        # Show as a table as well
        st.dataframe(selected_executives[['title', 'name', 'pay', 'currencyPay', 'gender', 'yearBorn']])
    else:
        st.write("No executive data available for this symbol.")

if __name__ == '__main__':
    main()
