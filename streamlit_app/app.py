import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# We'll use a Plotly qualitative color palette
from plotly.colors import qualitative as plotly_qual_colors

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

    # Merge
    df_merged = pd.merge(
        df_profiles, df_esg, 
        on='symbol', 
        how='left', 
        suffixes=('_profile', '_esg')
    )

    # Rename columns so they match your code
    df_merged.rename(columns={
        'companyName_profile': 'companyName',
        'fullTimeEmployees_profile': 'fullTimeEmployees',
        'industry_profile': 'industry',
    }, inplace=True)

    return df_profiles, df_esg, df_executives, raw_data_dict, df_merged


def main():
    st.set_page_config(page_title="Company & ESG Dashboard", layout="wide")
    st.title("Company & ESG Dashboard")

    st.write("""
    **This application** showcases data from multiple JSON files 
    (company profiles, executives, and ESG scores). 
    It includes a multi-tab interface for:
    - **Company Profile**, 
    - **ESG Analysis**, 
    - **Executives**, 
    - **Raw JSON**, and 
    - **Filtering & Clustering**.
    """)

    # Load data
    df_profiles, df_esg, df_executives, raw_data_dict, df_merged = load_data()

    # Sidebar filters for single-company selection
    st.sidebar.header("Symbol Selector")
    symbols_available = sorted(df_profiles['symbol'].dropna().unique())
    selected_symbol = st.sidebar.selectbox("Select a Symbol", symbols_available)

    # Filter for single symbol
    mask_profile = df_profiles['symbol'] == selected_symbol
    mask_esg = df_esg['symbol'] == selected_symbol
    mask_exec = df_executives['symbol'] == selected_symbol

    selected_profile = df_profiles[mask_profile].iloc[0] if not df_profiles[mask_profile].empty else None
    selected_esg = df_esg[mask_esg].iloc[0] if not df_esg[mask_esg].empty else None
    selected_executives = df_executives[mask_exec]
    selected_raw_json = raw_data_dict.get(selected_symbol, {})

    # Create Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "ESG Analysis",
        "Executives",
        "Raw JSON",
        "Filtering & Clustering",
    ])

    # ---------------- TAB 1: Overview ----------------
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

    # ---------------- TAB 2: ESG Analysis ----------------
    with tab2:
        st.subheader("ESG Analysis")
        if selected_esg is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Environmental", selected_esg.get('environmentalScore', 'N/A'))
            col2.metric("Social", selected_esg.get('socialScore', 'N/A'))
            col3.metric("Governance", selected_esg.get('governanceScore', 'N/A'))
            col4.metric("Overall ESG", selected_esg.get('ESGScore', 'N/A'))

            esg_scores_data = [
                {"Category": "Environmental", "Score": selected_esg.get('environmentalScore', None)},
                {"Category": "Social", "Score": selected_esg.get('socialScore', None)},
                {"Category": "Governance", "Score": selected_esg.get('governanceScore', None)},
            ]
            df_esg_scores = pd.DataFrame(esg_scores_data).dropna()

            if not df_esg_scores.empty:
                fig_esg_bar = px.bar(
                    df_esg_scores,
                    x="Category",
                    y="Score",
                    color="Category",
                    title="ESG Score Comparison",
                    range_y=[0, 100],
                )
                fig_esg_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_esg_bar, use_container_width=True)

            with st.expander("More ESG Details"):
                st.write(f"**Industry**: {selected_esg.get('industry', 'N/A')}")
                st.write(f"**Form Type**: {selected_esg.get('formType', 'N/A')}")
                st.write(f"**Date**: {selected_esg.get('date', 'N/A')}")
                st.write(f"**SEC Filing**: {selected_esg.get('url', 'N/A')}")
        else:
            st.warning("No ESG data available for this symbol.")

    # ---------------- TAB 3: Executives ----------------
    with tab3:
        st.subheader("Executives & Pay")
        if not selected_executives.empty:
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

            st.dataframe(
                selected_executives[['title', 'name', 'pay', 'currencyPay', 'gender', 'yearBorn']]
                .reset_index(drop=True)
            )
        else:
            st.warning("No executive data available for this symbol.")

    # ---------------- TAB 4: Raw JSON ----------------
    with tab4:
        st.subheader("Raw JSON Data")
        if selected_raw_json:
            st.write("Below is the raw JSON data for this symbol.")
            st.json(selected_raw_json)
        else:
            st.warning("No raw JSON found for this symbol.")

    # ---------------- TAB 5: Filtering & Clustering ----------------
    with tab5:
        st.subheader("Filtering & Clustering")

        st.markdown("""
        **Filter the full dataset** by various criteria (price, beta, exchange, sector, industry, city, state, 
        ESG scores, etc.), then run a simple **K-Means** clustering to see how these companies group together. 
        """)

        with st.sidebar.expander("Filtering Options", expanded=False):
            # Price range
            min_price, max_price = st.slider("Price Range", 0.0, 2000.0, (0.0, 2000.0), step=1.0)
            # Beta range
            min_beta, max_beta = st.slider("Beta Range", -5.0, 5.0, (-5.0, 5.0), step=0.1)
            # Employees range
            min_emp, max_emp = st.slider("Number of Employees Range", 0, 500000, (0, 500000), step=1000)

            # Exchange
            exchange_options = ["All"] + sorted([str(x) for x in df_merged['exchange'].dropna().unique()])
            selected_exchange = st.selectbox("Exchange", exchange_options)

            # Industry
            industry_options = ["All"] + sorted([str(x) for x in df_merged['industry_esg'].dropna().unique()])
            selected_industry = st.selectbox("Industry", industry_options)

            # Sector
            sector_options = ["All"] + sorted([str(x) for x in df_merged['sector'].dropna().unique()])
            selected_sector = st.selectbox("Sector", sector_options)

            # City
            city_options = ["All"] + sorted([str(x) for x in df_merged['city'].dropna().unique()])
            selected_city = st.selectbox("City", city_options)

            # State
            state_options = ["All"] + sorted([str(x) for x in df_merged['state'].dropna().unique()])
            selected_state = st.selectbox("State", state_options)

            # Environmental Score
            min_env, max_env = st.slider("Environmental Score Range", 0.0, 100.0, (0.0, 100.0), step=1.0)
            # Social Score
            min_soc, max_soc = st.slider("Social Score Range", 0.0, 100.0, (0.0, 100.0), step=1.0)
            # Governance Score
            min_gov, max_gov = st.slider("Governance Score Range", 0.0, 100.0, (0.0, 100.0), step=1.0)
            # Overall ESG Score
            min_esg, max_esg = st.slider("Overall ESG Score Range", 0.0, 100.0, (0.0, 100.0), step=1.0)

        df_filter = df_merged.copy()
        numeric_cols = [
            'price', 'beta', 'fullTimeEmployees',
            'environmentalScore', 'socialScore', 'governanceScore', 'ESGScore'
        ]
        for col in numeric_cols:
            df_filter[col] = pd.to_numeric(df_filter[col], errors='coerce')

        # Apply filters
        df_filter = df_filter[
            (df_filter['price'] >= min_price) &
            (df_filter['price'] <= max_price) &
            (df_filter['beta'] >= min_beta) &
            (df_filter['beta'] <= max_beta) &
            (df_filter['fullTimeEmployees'].fillna(0).astype(int) >= min_emp) &
            (df_filter['fullTimeEmployees'].fillna(0).astype(int) <= max_emp) &
            (df_filter['environmentalScore'] >= min_env) &
            (df_filter['environmentalScore'] <= max_env) &
            (df_filter['socialScore'] >= min_soc) &
            (df_filter['socialScore'] <= max_soc) &
            (df_filter['governanceScore'] >= min_gov) &
            (df_filter['governanceScore'] <= max_gov) &
            (df_filter['ESGScore'] >= min_esg) &
            (df_filter['ESGScore'] <= max_esg)
        ]

        if selected_exchange != "All":
            df_filter = df_filter[df_filter['exchange'] == selected_exchange]
        if selected_industry != "All":
            df_filter = df_filter[df_filter['industry_esg'] == selected_industry]
        if selected_sector != "All":
            df_filter = df_filter[df_filter['sector'] == selected_sector]
        if selected_city != "All":
            df_filter = df_filter[df_filter['city'] == selected_city]
        if selected_state != "All":
            df_filter = df_filter[df_filter['state'] == selected_state]

        st.write(f"**Filtered Dataset** has {len(df_filter)} rows (companies).")
        st.dataframe(df_filter[['symbol','companyName','price','beta','fullTimeEmployees',
                                'environmentalScore','socialScore','governanceScore','ESGScore']])

        st.markdown("---")
        st.markdown("### K-Means Clustering")

        st.write("""
        Choose the number of clusters and see how the filtered companies group together. 
        A simple PCA is used to reduce high-dimensional data into 2D for visualization.
        """)

        k = st.slider("Select Number of Clusters (K)", 2, 10, 3, step=1)
        
        cluster_cols = ['price', 'beta', 'fullTimeEmployees', 'environmentalScore', 'socialScore', 'governanceScore', 'ESGScore']
        df_cluster = df_filter[cluster_cols].dropna()

        if len(df_cluster) < k:
            st.warning("Not enough data points to form the selected number of clusters. Please adjust filters or choose fewer clusters.")
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_cluster)

            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)

            # Attach cluster labels back
            df_filter.loc[df_cluster.index, 'Cluster'] = labels

            # PCA for 2D
            pca = PCA(n_components=2, random_state=42)
            pcs = pca.fit_transform(X_scaled)

            df_plot = df_cluster.copy()
            df_plot['PC1'] = pcs[:, 0]
            df_plot['PC2'] = pcs[:, 1]
            df_plot['Cluster'] = labels
            df_plot['symbol'] = df_filter.loc[df_cluster.index, 'symbol']
            df_plot['companyName'] = df_filter.loc[df_cluster.index, 'companyName']

            # Convert cluster to string for discrete color
            df_plot['Cluster_str'] = df_plot['Cluster'].astype(str)
            
            # We'll pick a discrete color sequence (like Plotly or D3):
            color_sequence = plotly_qual_colors.Plotly  # 10 colors

            fig_cluster = px.scatter(
                df_plot,
                x='PC1', 
                y='PC2', 
                color='Cluster_str',
                hover_data=['symbol','companyName'],
                title=f"PCA Projection (K={k})",
                color_discrete_sequence=color_sequence
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

            st.dataframe(df_filter[['symbol','companyName','Cluster'] + cluster_cols])

            # --------------------------------------------------
            # Show each cluster with matching color
            # --------------------------------------------------
            unique_clusters = df_filter['Cluster'].dropna().unique()
            for cluster_label in unique_clusters:
                # We'll pick the color by indexing into color_sequence
                # (mod in case k>len(color_sequence), or define a bigger palette)
                cluster_color = color_sequence[int(cluster_label) % len(color_sequence)]
                
                # Subset for that cluster
                cluster_df = df_filter[df_filter['Cluster'] == cluster_label]
                
                # You can style the subheader with HTML:
                st.markdown(
                    f"<h3 style='color:{cluster_color};'>Cluster {int(cluster_label)} Results</h3>",
                    unsafe_allow_html=True
                )
                st.write(f"**Number of companies in Cluster {int(cluster_label)}:** {len(cluster_df)}")

                st.dataframe(cluster_df[['symbol','companyName','price','beta','fullTimeEmployees'] + 
                                        ['environmentalScore','socialScore','governanceScore','ESGScore']]
                )


if __name__ == "__main__":
    main()
