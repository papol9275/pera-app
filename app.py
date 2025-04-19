import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import time
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Financial Modeling Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for simpler UI
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: #2c3e50;
        border-radius: 0px;
        padding: 8px 16px;
        font-size: 14px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
    h1, h2, h3 {
        font-weight: 300;
    }
    .centered {
        text-align: center;
    }
    div[data-testid="stForm"] {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 4px;
        padding: 1em;
        margin-bottom: 1em;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 4px;
        height: 2.5em;
    }
    div[data-testid="stMetricValue"] {
        font-weight: bold;
    }
    .simple-header {
        border-bottom: 1px solid rgba(49, 51, 63, 0.2);
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("Financial Modeling Dashboard")
st.markdown("<p class='simple-header'>Comprehensive financial analysis and planning tools</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Navigation")
    st.info("Choose a financial model from the tabs in the main panel.")
    
    # Add other sidebar elements as needed
    st.subheader("About")
    st.write("This dashboard integrates various financial modeling tools for investment planning, loan analysis, and cash flow management.")
    
    # Date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.write(f"Updated: {current_time}")

# Create tabs for different models
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Investment Simulator", 
    "Cash Flow Analyzer", 
    "Loan Repayment", 
    "Personal Finance",
    "Basic Compounding",
    "USD Time Deposit"
])

# Tab 1: Investment Simulator (from Investing.py)
with tab1:
    st.header("Investment Growth Simulator")
    st.markdown("Simulate the growth of investments over time with customizable parameters.")
    
    with st.form("investment_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            years = st.number_input("Simulation Period (Years)", min_value=5, max_value=50, value=25, step=1)
            annual_investment = st.number_input("Annual Investment (₱)", min_value=0, value=250000, step=10000, format="%d")
            
        with col2:
            growth_rate = st.number_input("Annual Growth Rate (%)", min_value=1.0, max_value=20.0, value=3.75, step=0.25) / 100
            investment_duration = st.number_input("Investment Duration (Years)", min_value=1, max_value=years, value=min(30, years), step=1)
        
        submit_button = st.form_submit_button(label="Simulate Investment", use_container_width=True)
    
    # Define the simulation function (adapted from Investing.py)
    def simulate_investment(years, annual_investment, growth_rate, investment_duration):
        total_portfolio = 0
        total_invested = 0
        monthly_investment = 0
        withdrawals = 0
        cum_withdrawals = 0
        data = []  # List to hold yearly data for the DataFrame

        for year in range(1, years + 1):
            # Add the annual investment
            if year <= investment_duration:
                annual_invest = annual_investment
            else:
                annual_invest = 0
            
            total_portfolio += min(annual_invest * (1.04 ** (year - 1)), 1000000)
            total_invested += min(annual_invest * (1.04 ** (year - 1)), 1000000)
            monthly_investment = min(annual_invest * (1.04 ** (year - 1))/12, 1000000/12)

            if year > investment_duration:
                withdrawals = total_portfolio * growth_rate * 1

            # Calculate portfolio growth
            previous_portfolio = total_portfolio
            total_portfolio *= (1 + growth_rate)
            total_portfolio -= withdrawals
            growth_pesos = total_portfolio - previous_portfolio
            growth_percent = (growth_pesos / previous_portfolio) * 100
            cum_withdrawals += withdrawals
            
            # Append data for the current year
            data.append({
                "Year": year,
                "Age": year + 30,
                "Invested (₱)": total_invested,
                "Total Investment (₱)": previous_portfolio,
                "Year-end (₱)": total_portfolio,
                "Growth (₱)": growth_pesos,
                "Return (%)": growth_percent,
                "Monthly Contribution (₱)": monthly_investment,
                "Annual Contribution (₱)": monthly_investment * 12,
                "Withdrawals (₱)": withdrawals,
                "Total Withdrawals (₱)": cum_withdrawals
            })

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data)
        return df
    
    if submit_button or 'investment_df' not in st.session_state:
        # Run the simulation and store in session state
        result_df = simulate_investment(years, annual_investment, growth_rate, investment_duration)
        st.session_state.investment_df = result_df
    
    # Display results
    if 'investment_df' in st.session_state:
        df = st.session_state.investment_df
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Investment Value", f"₱{df['Year-end (₱)'].iloc[-1]:,.0f}")
        with col2:
            st.metric("Total Invested", f"₱{df['Invested (₱)'].iloc[-1]:,.0f}")
        with col3:
            st.metric("Total Returns", f"₱{(df['Year-end (₱)'].iloc[-1] - df['Invested (₱)'].iloc[-1]):,.0f}")
        with col4:
            st.metric("Total Withdrawals", f"₱{df['Total Withdrawals (₱)'].iloc[-1]:,.0f}")
        
        # Charts
        st.subheader("Investment Growth Over Time")
        tab_chart1, tab_chart2 = st.tabs(["Growth Chart", "Detailed Analysis"])
        
        with tab_chart1:
            fig = px.line(df, x="Year", y=["Year-end (₱)", "Invested (₱)"], 
                        title='Investment Growth Projection',
                        labels={"value": "Amount (₱)", "variable": "Category"},
                        color_discrete_map={"Year-end (₱)": "#4CAF50", "Invested (₱)": "#2196F3"})
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab_chart2:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.line(df, x="Year", y="Return (%)", title='Annual Return (%)',
                            labels={"Return (%)": "Return %", "Year": "Year"})
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.line(df, x="Year", y="Withdrawals (₱)", title='Annual Withdrawals',
                            labels={"Withdrawals (₱)": "Amount (₱)", "Year": "Year"})
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        # Data table with formatted values
        st.subheader("Detailed Investment Projection")
        display_df = df.copy()
        numeric_cols = ['Invested (₱)', 'Total Investment (₱)', 'Year-end (₱)', 'Growth (₱)', 
                        'Monthly Contribution (₱)', 'Annual Contribution (₱)', 'Withdrawals (₱)', 'Total Withdrawals (₱)']
        for col in numeric_cols:
            display_df[col] = display_df[col].apply(lambda x: f"₱{x:,.0f}")
        
        display_df['Return (%)'] = display_df['Return (%)'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(display_df, use_container_width=True)
        
        # Download button for the data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Investment Data as CSV", 
                          data=csv, 
                          file_name=f'investment_projection_{datetime.now().strftime("%Y%m%d")}.csv',
                          mime='text/csv')

# Tab 2: Cash Flow Analyzer (from Cashflow.py)
with tab2:
    st.header("Cash Flow Analyzer")
    st.markdown("Analyze cash flows with different repayment options over multiple years.")
    
    with st.form("cashflow_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            unit_price = st.number_input("Unit Price (₱)", min_value=10000, value=1000000, step=50000, format="%d")
            units_sold_annually = st.number_input("Units Sold Annually", min_value=1, value=8, step=1)
        
        with col2:
            years_to_simulate = st.number_input("Years to Simulate", min_value=1, max_value=30, value=3, step=1)
        
        submit_button = st.form_submit_button(label="Analyze Cash Flow", use_container_width=True)
    
    # Define the repayment options
    repayment_options = {
        "Option 1": {"months": 6, "monthly_interest": 0.0},
        "Option 2": {"months": 12, "monthly_interest": 0.0035},
        "Option 3": {"months": 18, "monthly_interest": 0.007},
        "Option 4": {"months": 24, "monthly_interest": 0.0115},
        "Option 5": {"months": 36, "monthly_interest": 0.0125},
    }
    
    # Function to calculate monthly payment (from Cashflow.py)
    def calculate_amortization(principal, monthly_rate, months):
        if monthly_rate == 0:  # Handle 0% interest
            return principal / months
        return principal * (monthly_rate * (1 + monthly_rate) ** months) / (
            (1 + monthly_rate) ** months - 1
        )

    # Function to simulate cash flows (adapted from Cashflow.py)
    def simulate_multiyear_cash_flows(unit_price, units_sold_annually, years_to_simulate, repayment_options):
        data = {}
        for option, details in repayment_options.items():
            months = details["months"]
            monthly_rate = details["monthly_interest"]
            monthly_payment = calculate_amortization(unit_price, monthly_rate, months)
            
            # Initialize cash flow tracker
            total_months = years_to_simulate * 12
            cash_flows = [0] * total_months
            
            # Add repayments for each year's sales
            for year in range(years_to_simulate):
                start_month = year * 12
                for i in range(months):
                    if start_month + i < total_months:
                        cash_flows[start_month + i] += monthly_payment * units_sold_annually
            
            # Group into annualized cash flows
            annual_cash_flows = {}
            for month, cash_flow in enumerate(cash_flows):
                year = month // 12 + 1
                annual_cash_flows[year] = annual_cash_flows.get(year, 0) + cash_flow
            
            # Cumulative cash flow
            cumulative_cash_flows = [sum(cash_flows[:i + 1]) for i in range(len(cash_flows))]
            
            data[option] = {
                "Monthly Payment": f"{monthly_payment:,.2f}",
                "Total Payment (Per Unit)": f"{monthly_payment * months:,.2f}",
                "Annualized Cash Flow": annual_cash_flows,
                "Cumulative Cash Flow": cumulative_cash_flows,
                "Monthly Payment Value": monthly_payment,
                "Total Payment Value": monthly_payment * months,
            }
        return data
    
    if submit_button or 'cashflow_data' not in st.session_state:
        # Run the simulation and store in session state
        cash_flow_data = simulate_multiyear_cash_flows(unit_price, units_sold_annually, years_to_simulate, repayment_options)
        st.session_state.cashflow_data = cash_flow_data
    
    # Display results
    if 'cashflow_data' in st.session_state:
        cash_flow_data = st.session_state.cashflow_data
        
        # Payment information
        st.subheader("Payment Options")
        
        # Create a comparison table of payment options
        payment_comparison = []
        for option, details in cash_flow_data.items():
            payment_comparison.append({
                "Option": option,
                "Duration (Months)": repayment_options[option]["months"],
                "Monthly Interest (%)": f"{repayment_options[option]['monthly_interest']*100:.2f}%",
                "Monthly Payment (₱)": details["Monthly Payment"],
                "Total Payment (₱)": details["Total Payment (Per Unit)"]
            })
        
        payment_df = pd.DataFrame(payment_comparison)
        st.dataframe(payment_df, use_container_width=True)
        
        # Create tabs for different views
        option_tab1, option_tab2 = st.tabs(["Annualized Cash Flow", "Cumulative Cash Flow"])
        
        with option_tab1:
            # Prepare data for annualized cash flow chart
            years = list(range(1, years_to_simulate + 1))
            annualized_data = {option: [] for option in cash_flow_data.keys()}
            
            for option, details in cash_flow_data.items():
                for year in years:
                    annualized_data[option].append(details["Annualized Cash Flow"].get(year, 0))
            
            # Create the annualized cash flow chart
            fig = go.Figure()
            for option, values in annualized_data.items():
                fig.add_trace(go.Bar(
                    x=years,
                    y=values,
                    name=f"{option} ({repayment_options[option]['months']} months)"
                ))
            
            fig.update_layout(
                title="Annualized Cash Flow by Repayment Option",
                xaxis_title="Year",
                yaxis_title="Cash Flow (₱)",
                barmode='group',
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table with annualized cash flow
            st.subheader("Annualized Cash Flow Data")
            annual_data = []
            for year in years:
                year_data = {"Year": year}
                for option in cash_flow_data.keys():
                    year_data[option] = f"₱{cash_flow_data[option]['Annualized Cash Flow'].get(year, 0):,.2f}"
                annual_data.append(year_data)
            
            annual_df = pd.DataFrame(annual_data)
            st.dataframe(annual_df, use_container_width=True)
        
        with option_tab2:
            # Prepare data for cumulative cash flow chart
            months = list(range(1, years_to_simulate * 12 + 1))
            
            # Create the cumulative cash flow chart
            fig = go.Figure()
            for option, details in cash_flow_data.items():
                fig.add_trace(go.Scatter(
                    x=months,
                    y=details["Cumulative Cash Flow"],
                    name=f"{option} ({repayment_options[option]['months']} months)",
                    mode='lines'
                ))
            
            fig.update_layout(
                title="Cumulative Cash Flow by Repayment Option",
                xaxis_title="Month",
                yaxis_title="Cumulative Cash Flow (₱)",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly detail view (toggleable)
            if st.checkbox("Show Monthly Detail", value=False):
                selected_option = st.selectbox(
                    "Select Payment Option to View Monthly Detail", 
                    list(cash_flow_data.keys())
                )
                
                months_data = []
                for i, value in enumerate(cash_flow_data[selected_option]["Cumulative Cash Flow"]):
                    months_data.append({
                        "Month": i + 1,
                        "Monthly Cash Flow": cash_flow_data[selected_option]["Cumulative Cash Flow"][i] - 
                                           (cash_flow_data[selected_option]["Cumulative Cash Flow"][i-1] if i > 0 else 0),
                        "Cumulative Cash Flow": value
                    })
                
                months_df = pd.DataFrame(months_data)
                # Format currency values
                months_df["Monthly Cash Flow"] = months_df["Monthly Cash Flow"].apply(lambda x: f"₱{x:,.2f}")
                months_df["Cumulative Cash Flow"] = months_df["Cumulative Cash Flow"].apply(lambda x: f"₱{x:,.2f}")
                
                st.dataframe(months_df, use_container_width=True)
        
        # Download button for the data
        st.subheader("Export Data")
        
        # Prepare data for export
        export_data = []
        for year in range(1, years_to_simulate + 1):
            year_data = {"Year": year}
            for option in cash_flow_data.keys():
                year_data[f"{option} Annual Cash Flow"] = cash_flow_data[option]["Annualized Cash Flow"].get(year, 0)
                year_data[f"{option} Monthly Payoff"] = cash_flow_data[option]["Monthly Payment Value"]
                year_data[f"{option} Total Payment"] = cash_flow_data[option]["Total Payment Value"]
            export_data.append(year_data)
        
        export_df = pd.DataFrame(export_data)
        csv = export_df.to_csv(index=False).encode('utf-8')
        
        st.download_button("Download Cash Flow Data as CSV", 
                          data=csv, 
                          file_name=f'cash_flow_analysis_{datetime.now().strftime("%Y%m%d")}.csv',
                          mime='text/csv')

# Tab 3: Loan Repayment (from Loan Repayment.py)
with tab3:
    st.header("Loan Repayment Calculator")
    st.markdown("Calculate loan repayment schedules and analyze interest payments.")
    
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            loan_amount = st.number_input("Loan Amount (₱)", min_value=1000, value=100000, step=5000, format="%d")
            carryover_amount = st.number_input("Carryover Amount (₱)", min_value=0, value=10000, step=1000, format="%d")
        
        with col2:
            monthly_interest_rate = st.number_input("Monthly Interest Rate (%)", min_value=0.0, value=1.50, step=0.25, format="%.2f")
            carryover_duration = st.number_input("Carryover Duration (Months)", min_value=0, value=12, step=1)
            duration_months = st.number_input("Repayment Duration (Months)", min_value=1, value=36, step=6)
            
        submit_button = st.form_submit_button(label="Calculate Repayment Schedule", use_container_width=True)
    
    # Function to calculate amortized payment
    def calculate_amortized_payment(principal, monthly_rate, months_remaining):
        """Calculates the amortized monthly payment."""
        if monthly_rate == 0:  # Handle zero interest rate
            return principal / months_remaining
        return (principal * (monthly_rate * (1 + monthly_rate) ** months_remaining) / (
            (1 + monthly_rate) ** months_remaining - 1)
        )
    
    # Function to calculate repayment schedule
    def calculate_repayment_schedule(loan_amount, monthly_interest_rate, duration_months, carryover_amount, carryover_duration):
        monthly_rate = monthly_interest_rate / 100
        balance = loan_amount
        equity_balance = ((loan_amount - 0.12 * loan_amount / 1.12) - (loan_amount - 0.12 * loan_amount / 1.12) * (0.25/1.25)) * -1
        capital = (loan_amount - 0.12 * loan_amount / 1.12) - (loan_amount - 0.12 * loan_amount / 1.12)*(0.25/1.25)
        markup = (loan_amount - 0.12 * loan_amount / 1.12) * (0.25/1.25)
        cumulative_interest = 0
        repayment_schedule = []
        return_on_capital = equity_balance
        repayment = 0
        tax = 0
        net_pnl = 0
        
        # Calculate monthly carryover payment
        carryover_payment = carryover_amount if carryover_duration == 0 else carryover_amount * carryover_duration / duration_months

        month = 1
        while month <= duration_months:
            interest = balance * monthly_rate
            payment = calculate_amortized_payment(balance, monthly_rate, duration_months - month + 1)
            principal_payment = payment - interest
            ending_balance = balance - principal_payment
            cumulative_interest += interest
            effective_interest_percentage = (cumulative_interest / loan_amount) * 100
            repayment += payment
            tax += payment * 0.12 / 1.12
            return_on_capital = ((repayment - tax + equity_balance))
            net_pnl = repayment - tax + equity_balance
            roi = -100 * net_pnl / equity_balance if equity_balance != 0 else 0
            total_repayment = payment + carryover_payment

            repayment_schedule.append({
                "Month": month,
                "Beginning Balance": balance,
                "Interest": interest,
                "Loan Payment": payment,
                "Carryover Payment": carryover_payment,
                "Total Repayment": total_repayment,
                "Ending Balance": ending_balance,
                "Cumulative Interest": cumulative_interest,
                "Interest Percentage": effective_interest_percentage,
                "ROI": roi,
                "Net PnL": net_pnl
            })
            
            balance = ending_balance
            month += 1
            
        return pd.DataFrame(repayment_schedule)
    
    if submit_button or 'loan_df' not in st.session_state:
        # Calculate the repayment schedule and store in session state
        loan_df = calculate_repayment_schedule(loan_amount, monthly_interest_rate, duration_months, carryover_amount, carryover_duration)
        st.session_state.loan_df = loan_df
    
    # Display results
    if 'loan_df' in st.session_state:
        df = st.session_state.loan_df
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Payments", f"₱{df['Total Repayment'].sum():,.2f}")
        with col2:
            st.metric("Total Interest", f"₱{df['Interest'].sum():,.2f}")
        with col3:
            st.metric("Interest Percentage", f"{df['Interest Percentage'].iloc[-1]:.2f}%")
        with col4:
            st.metric("Monthly Payment", f"₱{df['Loan Payment'].iloc[0]:,.2f}")
        
        # Charts
        st.subheader("Loan Amortization Charts")
        chart_tab1, chart_tab2 = st.tabs(["Balance & Interest", "ROI Analysis"])
        
        with chart_tab1:
            # Create a plot with two y-axes
            fig = px.line(df, x="Month", y=["Beginning Balance", "Cumulative Interest"], 
                       title='Balance and Cumulative Interest Over Time',
                       labels={"value": "Amount (₱)", "variable": "Category"},
                       color_discrete_map={"Beginning Balance": "#1E88E5", "Cumulative Interest": "#FF5722"})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Payment breakdown chart
            payment_data = df[["Month", "Interest", "Loan Payment"]].copy()
            payment_data["Principal"] = payment_data["Loan Payment"] - payment_data["Interest"]
            
            fig = px.bar(payment_data, x="Month", y=["Principal", "Interest"],
                       title='Payment Breakdown - Principal vs Interest',
                       labels={"value": "Amount (₱)", "variable": "Payment Component"},
                       color_discrete_map={"Principal": "#4CAF50", "Interest": "#FF5722"})
            fig.update_layout(height=400, barmode='stack')
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_tab2:
            # ROI and Net PnL over time
            fig = px.line(df, x="Month", y=["ROI", "Net PnL"], 
                       title='ROI and Net Profit/Loss Over Time',
                       labels={"value": "Amount", "variable": "Metric"},
                       color_discrete_map={"ROI": "#673AB7", "Net PnL": "#2196F3"})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Data table with formatted values
        st.subheader("Detailed Repayment Schedule")
        display_df = df.copy()
        
        # Format currency values
        currency_cols = ['Beginning Balance', 'Interest', 'Loan Payment', 'Carryover Payment', 
                        'Total Repayment', 'Ending Balance', 'Cumulative Interest', 'Net PnL']
        for col in currency_cols:
            display_df[col] = display_df[col].apply(lambda x: f"₱{x:,.2f}")
        
        # Format percentage values
        display_df['Interest Percentage'] = display_df['Interest Percentage'].apply(lambda x: f"{x:.2f}%")
        display_df['ROI'] = display_df['ROI'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button for the data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Repayment Schedule as CSV", 
                          data=csv, 
                          file_name=f'loan_repayment_schedule_{datetime.now().strftime("%Y%m%d")}.csv',
                          mime='text/csv')

# Tab 4: Multi-Loan Manager (from Multiloan Repayment.py)
with tab4:
    st.header("Personal Finance Management")
    st.markdown("Manage and analyze multiple loans with different terms.")
    
    # Initialize session state for number of loans if not present
    if 'num_loans' not in st.session_state:
        st.session_state.num_loans = 3
    
    # Add a control to adjust the number of loans outside the form
    loan_count_col1, loan_count_col2 = st.columns([3, 1])
    with loan_count_col1:
        new_num_loans = st.number_input("Number of loans to manage", min_value=1, max_value=10, value=st.session_state.num_loans, step=1)
    with loan_count_col2:
        if st.button("Update Loan Count"):
            st.session_state.num_loans = new_num_loans
            # Reset the session state to force recomputation with the new loan count
            if 'multiloan_df' in st.session_state:
                del st.session_state.multiloan_df
            st.rerun()
    
    with st.form("multiloan_form"):
        st.subheader("Loan Details")
        
        # Dynamically create loan input fields based on number of loans
        loan_amounts = []
        carry_amounts = []
        carry_durations = []
        
        # Create loans in rows of 3 for better UI organization
        for i in range(0, st.session_state.num_loans, 3):
            loan_cols = st.columns(3)
            
            # Add up to 3 loans per row
            for j in range(min(3, st.session_state.num_loans - i)):
                loan_idx = i + j
                with loan_cols[j]:
                    st.markdown(f"### Loan {loan_idx + 1}")
                    amount = st.number_input(
                        f"Loan {loan_idx + 1} Amount (₱)", 
                        min_value=0, 
                        value=5000 if loan_idx == 0 else (5000 if loan_idx == 1 else 5000), 
                        step=5000, 
                        format="%d",
                        key=f"loan_amount_{loan_idx}"
                    )
                    carry = st.number_input(
                        f"Carryover {loan_idx + 1} Amount (₱)", 
                        min_value=0, 
                        value=5000 if loan_idx == 0 else (5000 if loan_idx == 1 else 5000), 
                        step=5000, 
                        format="%d",
                        key=f"carry_amount_{loan_idx}"
                    )
                    duration = st.number_input(
                        f"Carryover {loan_idx + 1} Duration (Months)", 
                        min_value=0, 
                        value=16, 
                        step=1,
                        key=f"carry_duration_{loan_idx}"
                    )
                    loan_amounts.append(amount)
                    carry_amounts.append(carry)
                    carry_durations.append(duration)
        
        # Common parameters
        st.markdown("### Common Parameters")
        
        common_cols = st.columns(2)
        with common_cols[0]:
            monthly_interest = st.number_input("Monthly Interest Rate (%)", min_value=0.0, value=3.0, step=0.5, format="%.2f")
        with common_cols[1]:
            monthly_budget = st.number_input("Monthly Payoff Budget (₱)", min_value=5000, value=10000, step=5000, format="%d")
        
        # Add monthly income and expenses configuration
        st.markdown("### Monthly Income & Expenses")
        income_cols = st.columns(3)
        with income_cols[0]:
            monthly_income = st.number_input("Monthly Income (₱)", min_value=0, value=50000, step=500, format="%d", 
                                           help="Your total monthly income from all sources")
        with income_cols[1]:
            monthly_expenses = st.number_input("Monthly Expenses (₱)", min_value=0, value=25000, step=500, format="%d",
                                             help="Your fixed monthly expenses excluding loan payments")
        with income_cols[2]:
            disposable_income = st.number_input("Disposable (₱)", min_value=0, value=10000, step=500, format="%d",
                                             help="Your fixed monthly disposables")
        # Additional parameters
        cont_months = st.number_input("Trailing Months After Loan Payoff", min_value=0, value=12, step=1,
                                    help="Number of months to continue financial tracking after all loans are paid off")
        
        submit_button = st.form_submit_button(label="Calculate Multi-Loan Repayment", use_container_width=True)
    
    # Function to calculate multi-loan repayment - updated to handle any number of loans
    def calculate_multiloan_repayment(loan_amounts, carry_amounts, carry_durations, monthly_rate, budget, 
                                      monthly_income, monthly_expenses, cont_months):
        monthly_rate = monthly_rate / 100
        loan_balances = loan_amounts.copy()
        num_loans = len(loan_amounts)
        
        schedule = []
        cumulative_savings = []
        month = 1
        all_paid_off_month = None  # Track when all loans are paid off
        max_iterations = 1000  # Safety limit to prevent infinite loops
        
        # Create copies of carryover durations to avoid modifying the original
        carry_durations_remaining = carry_durations.copy()
        
        # Continue until all loans are paid off and carryover periods are complete,
        # plus the continuation period requested
        while month < max_iterations:  # Add a safety limit
            # Check if we're in post-payoff continuation period
            loans_active = any(b > 0 for b in loan_balances)
            carryovers_active = any(c > 0 for c in carry_durations_remaining)
            
            # If this is the first month where everything is paid off, mark it
            if not loans_active and not carryovers_active and all_paid_off_month is None:
                all_paid_off_month = month
            
            # Determine if we should continue or exit
            if not loans_active and not carryovers_active:
                # If we've already extended beyond the continuation period, exit
                if all_paid_off_month is not None and month >= all_paid_off_month + cont_months:
                    break
            
            # Add carryover amounts (and decrement duration) - only if carryovers are still active
            carry_pays = [0] * num_loans
            leftover = budget
            
            for i in range(num_loans):
                if carry_durations_remaining[i] > 0:
                    pay_carry = min(carry_amounts[i], leftover)
                    carry_pays[i] = pay_carry
                    leftover -= pay_carry
                    carry_durations_remaining[i] -= 1
            
            # Calculate interest - only on active loans
            interests = [(b * monthly_rate) if b > 0 else 0 for b in loan_balances]
            total_interest = sum(interests)
            
            # Allocate leftover budget to highest interest first 
            pay_allocation = [0] * num_loans
            if any(b > 0 for b in loan_balances):  # Only allocate if there are active loans
                # Sort loans by interest (highest first)
                order = sorted(range(num_loans), key=lambda x: interests[x] if loan_balances[x] > 0 else -1, reverse=True)
                pay_left = leftover
                for i in order:
                    if loan_balances[i] > 0:  # Only allocate to loans with remaining balance
                        # Pay at least the interest, then apply remaining to principal
                        interest_payment = min(interests[i], pay_left)
                        principal_payment = min(loan_balances[i], pay_left - interest_payment)
                        pay_allocation[i] = interest_payment + principal_payment
                        pay_left -= pay_allocation[i]
            
            # Update balances
            new_balances = []
            for i in range(num_loans):
                if loan_balances[i] > 0:
                    # Calculate principal payment (payment minus interest)
                    principal_payment = max(0, pay_allocation[i] - interests[i])
                    # Ensure we don't overpay the loan
                    principal_payment = min(principal_payment, loan_balances[i])
                    # Calculate new balance
                    new_bal = loan_balances[i] - principal_payment
                    new_balances.append(max(0, new_bal))  # Ensure balance is never negative
                else:
                    new_balances.append(0)
            
            # Calculate monthly financials
            actual_loan_payment = sum(carry_pays) + sum(pay_allocation)

            # Increase monthly income by 1.5% for each passing year
            if month % 12 == 0:
                monthly_income *= 1.05
                monthly_expenses *= 1.05
            
            # Actual savings is disposable income minus what was spent on loans
            monthly_savings = 2 * monthly_income - monthly_expenses - actual_loan_payment - disposable_income if month % 12 == 0 else monthly_income - monthly_expenses - actual_loan_payment - disposable_income
            
            # Track cumulative savings
            current_cumulative_savings = cumulative_savings[-1] + monthly_savings if cumulative_savings else monthly_savings
            cumulative_savings.append(current_cumulative_savings)
            
            # Build row data - dynamically based on number of loans and financial status
            row = {
                "Year": (month - 1) // 12 + 1,
                "Month": month,
                "Total Interest": total_interest,
                "Total Payment": actual_loan_payment,
                "Monthly Income": 2 * monthly_income if month % 12 == 0 else monthly_income,
                "Monthly Expenses": monthly_expenses,
                "Disposable Income": disposable_income,
                "Monthly Savings": monthly_savings,
                "Cumulative Savings": current_cumulative_savings,
                "Financial Status": "Post-Payoff" if all_paid_off_month is not None else "Active Loans"
            }
            
            # Add loan-specific columns
            for i in range(num_loans):
                row[f"Loan{i+1} Balance"] = loan_balances[i]
                row[f"Loan{i+1} Payment"] = pay_allocation[i]
                row[f"Interest{i+1}"] = interests[i]
                if carry_durations_remaining[i] > 0:
                    row[f"Carryover{i+1} Payment"] = carry_pays[i]
                    row[f"Carryover{i+1} Remaining"] = carry_durations_remaining[i]
            
            schedule.append(row)
            
            month += 1
            loan_balances = new_balances
            
            # Additional safety check - if all balances are zero and we've gone past the minimum
            # months specified by carry_durations + cont_months, we can exit
            if all(b == 0 for b in loan_balances) and all(c == 0 for c in carry_durations_remaining) and month > max(carry_durations or [0]) + cont_months:
                # Only exit if we've already processed at least the minimum expected months
                if month > 12:  # Ensure we show at least a year's worth of data
                    break
        
        # Add a summary row to indicate when all loans are paid off
        if all_paid_off_month is not None:
            payoff_info = f"All loans paid off at month {all_paid_off_month}"
        else:
            payoff_info = "Loans not fully paid off within maximum simulation period"
        
        result_df = pd.DataFrame(schedule)
        
        # Add metadata as an attribute to the DataFrame
        result_df.attrs['payoff_month'] = all_paid_off_month
        result_df.attrs['payoff_info'] = payoff_info
        
        return result_df
    
    if submit_button or 'multiloan_df' not in st.session_state:
        # Calculate the multi-loan repayment schedule and store in session state
        multiloan_df = calculate_multiloan_repayment(
            loan_amounts=loan_amounts, 
            carry_amounts=carry_amounts, 
            carry_durations=carry_durations, 
            monthly_rate=monthly_interest, 
            budget=monthly_budget,
            monthly_income=monthly_income,
            monthly_expenses=monthly_expenses,
            cont_months=cont_months
        )
        st.session_state.multiloan_df = multiloan_df
    
    # Display results
    if 'multiloan_df' in st.session_state:
        df = st.session_state.multiloan_df
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_months = len(df)
            st.metric("Total Months to Payoff", f"{total_months}")
        with col2:
            total_interest = df['Total Interest'].sum()
            st.metric("Total Interest Paid", f"₱{total_interest:,.2f}")
        with col3:
            total_payments = df['Total Payment'].sum()
            st.metric("Total Payments", f"₱{total_payments:,.2f}")
        with col4:
            total_savings = df['Cumulative Savings'].iloc[-1]
            st.metric("Total Savings", f"₱{total_savings:,.2f}")
        
        # Charts
        st.subheader("Multi-Loan Analysis Charts")
        chart_tab1, chart_tab2 = st.tabs(["Loan Balances", "Payments & Savings"])
        
        with chart_tab1:
            # Combine loan balances - handle any number of loans dynamically
            loan_balance_cols = [col for col in df.columns if col.startswith('Loan') and col.endswith('Balance')]
            df['Total Balance'] = df[loan_balance_cols].sum(axis=1)
            
            # Create a color map for any number of loans
            color_options = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#673AB7", "#00BCD4", "#FFC107", "#795548", "#607D8B", "#9C27B0"]
            color_map = {loan_balance_cols[i]: color_options[i % len(color_options)] for i in range(len(loan_balance_cols))}
            color_map["Total Balance"] = "#673AB7"  # Always use purple for total
            
            # Create a plot for loan balances
            fig = px.line(df, x="Month", 
                       y=loan_balance_cols + ["Total Balance"],
                       title='Loan Balances Over Time',
                       labels={"value": "Balance (₱)", "variable": "Loan"},
                       color_discrete_map=color_map)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_tab2:
            # Create plots for payments and savings - handle any number of loans dynamically
            payment_cols = [col for col in df.columns if col.startswith('Loan') and col.endswith('Payment')]
            
            # Create a color map for any number of loans
            color_options = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#673AB7", "#00BCD4", "#FFC107", "#795548", "#607D8B", "#9C27B0"]
            payment_color_map = {payment_cols[i]: color_options[i % len(color_options)] for i in range(len(payment_cols))}
            
            fig = px.area(df, x="Month", y=payment_cols,
                        title='Monthly Payments by Loan',
                        labels={"value": "Payment (₱)", "variable": "Loan"},
                        color_discrete_map=payment_color_map)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Savings over time
            fig = px.line(df, x="Month", y=["Monthly Savings", "Cumulative Savings"],
                        title='Monthly and Cumulative Savings',
                        labels={"value": "Amount (₱)", "variable": "Savings Type"},
                        color_discrete_map={
                            "Monthly Savings": "#009688", 
                            "Cumulative Savings": "#E91E63"
                        })
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Split the data into two separate dataframes for better insights
        
        # Personal Finance Summary dataframe
        st.subheader("Personal Finance Summary")
        finance_summary_cols = ["Year", "Month", "Monthly Income", "Monthly Expenses", "Total Payment", 
                               "Disposable Income", "Monthly Savings", "Cumulative Savings", "Financial Status"]
        finance_summary_df = df[finance_summary_cols].copy()
        
        # Format currency values for finance summary
        numeric_cols = finance_summary_df.select_dtypes(include=['float', 'int']).columns.tolist()
        currency_cols = [col for col in numeric_cols if col not in ['Month', 'Year']]
        
        for col in currency_cols:
            finance_summary_df[col] = finance_summary_df[col].apply(lambda x: f"₱{x:,.2f}" if isinstance(x, (int, float)) else x)
        
        st.dataframe(finance_summary_df, use_container_width=True)
        
        # Loan Payment Summary dataframe
        st.subheader("Loan Payment Summary")
        
        # Get all loan payment, carryover payment, and balance columns dynamically
        loan_payment_cols = [col for col in df.columns if col.startswith('Loan') and col.endswith('Payment')]
        carry_payment_cols = [col for col in df.columns if col.startswith('Carryover') and col.endswith('Payment')]
        loan_balance_cols = [col for col in df.columns if col.startswith('Loan') and col.endswith('Balance')]
        
        # Create a temporary dataframe to combine loan and carryover payments
        temp_df = df.copy()
        
        # Create combined payment columns (loan payment + carryover payment)
        combined_payment_cols = []
        for i in range(len(loan_payment_cols)):
            loan_col = f"Loan{i+1} Payment"
            carry_col = f"Carryover{i+1} Payment"
            
            combined_col = f"Total Loan {i+1} Payment"
            combined_payment_cols.append(combined_col)
            
            # Add loan payment and carryover payment (handle missing columns)
            if loan_col in temp_df.columns and carry_col in temp_df.columns:
                temp_df[combined_col] = temp_df[loan_col] + temp_df[carry_col]
            elif loan_col in temp_df.columns:
                temp_df[combined_col] = temp_df[loan_col]
            elif carry_col in temp_df.columns:
                temp_df[combined_col] = temp_df[carry_col]
            else:
                temp_df[combined_col] = 0
        
        # Create the payment summary columns list - keep it focused on key loan metrics
        payment_summary_cols = ["Month"] + combined_payment_cols + ["Total Interest", "Total Balance"]
        
        # Add Total Balance to the dataframe if it doesn't exist
        if "Total Balance" not in temp_df.columns:
            temp_df["Total Balance"] = temp_df[loan_balance_cols].sum(axis=1)
            
        payment_summary_df = temp_df[payment_summary_cols].copy()
        
        # Format currency values for payment summary
        numeric_cols = payment_summary_df.select_dtypes(include=['float', 'int']).columns.tolist()
        currency_cols = [col for col in numeric_cols if col != 'Month']
        
        for col in currency_cols:
            payment_summary_df[col] = payment_summary_df[col].apply(lambda x: f"₱{x:,.2f}" if isinstance(x, (int, float)) else x)
        
        # Calculate and add a "Total Monthly Payment" column (sum of loan payments and carryover payments)
        loan_cols = [col for col in payment_summary_df.columns if col.startswith('Loan') and col.endswith('Payment')]
        carry_cols = [col for col in payment_summary_df.columns if col.startswith('Carryover') and col.endswith('Payment')]
        
        # Need to convert back to numeric before summing
        temp_df = df.copy()
        payment_cols = loan_payment_cols + carry_payment_cols
        total_monthly_payments = temp_df[payment_cols].sum(axis=1)
        
        # Add to the display dataframe and format
        payment_summary_df.insert(len(payment_summary_df.columns)-2, "Total Monthly Payment", 
                               total_monthly_payments.apply(lambda x: f"₱{x:,.2f}" if isinstance(x, (int, float)) else x))
        
        st.dataframe(payment_summary_df, use_container_width=True)
        
        # Download button for the data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Multi-Loan Schedule as CSV", 
                          data=csv, 
                          file_name=f'multiloan_repayment_schedule_{datetime.now().strftime("%Y%m%d")}.csv',
                          mime='text/csv')

# Tab 5: Basic Compounding (from Basic Compounding.py)
with tab5:
    st.header("Basic Compounding Calculator")
    st.markdown("Visualize the power of compound growth over time.")
    
    with st.form("compounding_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            initial_investment = st.number_input("Initial Investment (₱)", min_value=1000, value=5000, step=1000, format="%d")
            annual_contribution = st.number_input("Annual Contribution (₱)", min_value=0, value=1000, step=100, format="%d")
        
        with col2:
            growth_rate = st.number_input("Annual Growth Rate (%)", min_value=1.0, max_value=30.0, value=6.0, step=0.5) / 100
            years = st.number_input("Number of Years", min_value=1, max_value=100, value=30, step=1)
        
        submit_button = st.form_submit_button(label="Calculate Compound Growth", use_container_width=True)
    
    # Function to simulate compounding (adapted from Basic Compounding.py)
    def simulate_basic_compounding(years, initial_investment, annual_contribution, growth_rate):
        total_portfolio = initial_investment
        total_invested = initial_investment
        data = []  # List to hold yearly data for the DataFrame

        for year in range(1, years + 1):
            # Add the annual contribution
            total_portfolio += annual_contribution
            total_invested += annual_contribution
            
            # Calculate portfolio growth
            previous_portfolio = total_portfolio
            total_portfolio *= (1 + growth_rate)
            growth_pesos = total_portfolio - previous_portfolio
            growth_percent = (growth_pesos / total_invested) * 100
            
            # Append data for the current year
            data.append({
                "Year": year,
                "Days": year * 365,
                "Invested (₱)": total_invested,
                "Total Investment (₱)": previous_portfolio,
                "Ending (₱)": total_portfolio,
                "Growth (₱)": growth_pesos,
                "Growth (%)": growth_percent
            })

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data)
        return df
    
    if submit_button or 'compounding_df' not in st.session_state:
        # Calculate the compounding growth and store in session state
        compounding_df = simulate_basic_compounding(years, initial_investment, annual_contribution, growth_rate)
        st.session_state.compounding_df = compounding_df
    
    # Display results
    if 'compounding_df' in st.session_state:
        df = st.session_state.compounding_df
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            final_value = df['Ending (₱)'].iloc[-1]
            st.metric("Final Value", f"₱{final_value:,.0f}")
        with col2:
            total_invested = df['Invested (₱)'].iloc[-1]
            st.metric("Total Invested", f"₱{total_invested:,.0f}")
        with col3:
            total_growth = final_value - total_invested
            st.metric("Total Growth", f"₱{total_growth:,.0f}")
        with col4:
            growth_multiple = final_value / total_invested
            st.metric("Growth Multiple", f"{growth_multiple:.2f}x")
        
        # Charts
        st.subheader("Compound Growth Visualization")
        
        # Chart type selection
        chart_type = st.radio("Select Chart Type", ["Linear", "Logarithmic"], horizontal=True)
        
        if chart_type == "Linear":
            # Linear scale chart
            fig = px.line(df, x="Year", y=["Ending (₱)", "Invested (₱)"],
                        title='Compound Growth Over Time (Linear Scale)',
                        labels={"value": "Amount (₱)", "variable": "Category"},
                        color_discrete_map={"Ending (₱)": "#4CAF50", "Invested (₱)": "#2196F3"})
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Logarithmic scale chart
            fig = px.line(df, x="Year", y=["Ending (₱)", "Invested (₱)"],
                        title='Compound Growth Over Time (Logarithmic Scale)',
                        labels={"value": "Amount (₱)", "variable": "Category"},
                        color_discrete_map={"Ending (₱)": "#4CAF50", "Invested (₱)": "#2196F3"},
                        log_y=True)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Growth percentage over time
        fig = px.line(df, x="Year", y="Growth (%)",
                    title='Annual Growth Percentage',
                    labels={"Growth (%)": "Growth (%)", "Year": "Year"},
                    color_discrete_sequence=["#FF5722"])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table with formatted values
        st.subheader("Detailed Growth Projection")
        display_df = df.copy()
        
        # Format currency values
        currency_cols = ['Invested (₱)', 'Total Investment (₱)', 'Ending (₱)', 'Growth (₱)']
        for col in currency_cols:
            display_df[col] = display_df[col].apply(lambda x: f"₱{x:,.0f}")
        
        # Format percentage values
        display_df['Growth (%)'] = display_df['Growth (%)'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button for the data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Compound Growth Data as CSV", 
                          data=csv, 
                          file_name=f'compound_growth_projection_{datetime.now().strftime("%Y%m%d")}.csv',
                          mime='text/csv')

# Tab 6: USD Time Deposit Investment Scheme
with tab6:
    st.header("USD Time Deposit Investment Scheme")
    st.markdown("Simulate investing in USD time deposits with monthly investments and 3-month maturity periods.")
    
    # Import the fixed time deposit function with proper compounding
    from fixed_time_deposit import simulate_usd_time_deposit_monthly
    
    with st.form("usd_deposit_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_peso_investment = st.number_input("Monthly Investment (₱)", min_value=1000, value=10000, step=2500, format="%d")
            initial_usd_value = st.number_input("Initial USD-PHP Exchange Rate", min_value=40.0, value=56.5, step=0.5, format="%.2f")
        
        with col2:
            investment_years = st.number_input("Investment Period (Years)", min_value=1, max_value=45, value=5, step=1)
            show_monthly = st.checkbox("Show Monthly Breakdown", value=False)
        
        submit_button = st.form_submit_button(label="Calculate USD Time Deposit Growth", use_container_width=True)
    
    if submit_button or 'usd_deposit_df' not in st.session_state:
        # Run the simulation with the new function and store in session state
        result_df = simulate_usd_time_deposit_monthly(
            monthly_peso_investment, 
            initial_usd_value, 
            investment_years, 
            show_monthly
        )
        st.session_state.usd_deposit_df = result_df
    
    # Display results
    if 'usd_deposit_df' in st.session_state:
        df = st.session_state.usd_deposit_df
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            final_usd = df['Total USD Holdings'].iloc[-1]
            st.metric("Final USD Holdings", f"${final_usd:,.2f}")
        with col2:
            final_peso_value = df['Holdings Value (₱)'].iloc[-1]
            st.metric("Final Peso Value", f"₱{final_peso_value:,.2f}")
        with col3:
            total_invested = df['Total Invested (₱)'].iloc[-1]
            st.metric("Total Invested", f"₱{total_invested:,.2f}")
        with col4:
            final_roi = df['ROI (%)'].iloc[-1]
            st.metric("Total ROI", f"{final_roi:.2f}%")
        
        # Charts
        st.subheader("USD Time Deposit Growth")
        
        # Create tabs for different views
        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Growth Chart", "Exchange Rate & Returns", "Maturity & Reinvestment"])
        
        with chart_tab1:
            # Plot USD holdings and Peso value over time
            fig = go.Figure()
            
            # USD Holdings line
            fig.add_trace(go.Scatter(
                x=df['Period'],
                y=df['Total USD Holdings'],
                name="USD Holdings",
                line=dict(color='#2196F3', width=2),
                yaxis='y'
            ))
            
            # Peso Value line
            fig.add_trace(go.Scatter(
                x=df['Period'],
                y=df['Holdings Value (₱)'],
                name="Peso Value",
                line=dict(color='#4CAF50', width=2, dash='dash'),
                yaxis='y2'
            ))
            
            # Set up dual y-axes
            fig.update_layout(
                title='USD Holdings and Peso Value Over Time',
                yaxis=dict(
                    title=dict(text="USD Holdings ($)", font=dict(color='#2196F3')),
                    tickfont=dict(color='#2196F3')
                ),
                yaxis2=dict(
                    title=dict(text="Value in Pesos (₱)", font=dict(color='#4CAF50')),
                    tickfont=dict(color='#4CAF50'),
                    anchor="x",
                    overlaying="y",
                    side="right"
                ),
                xaxis=dict(
                    title="Time Period"
                ),
                height=500,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Purchased vs Interest Earned stacked bar chart
            if 'Month' not in df.columns:  # Only for quarterly/yearly view for clarity
                fig = go.Figure()
                
                # Add USD Purchased bars
                fig.add_trace(go.Bar(
                    x=df['Period'],
                    y=df['USD Purchased'],
                    name="USD Purchased",
                    marker_color='#FF9800'
                ))
                
                # Add Interest Earned bars
                fig.add_trace(go.Bar(
                    x=df['Period'],
                    y=df['Interest Earned ($)'],
                    name="Interest Earned",
                    marker_color='#9C27B0'
                ))
                
                fig.update_layout(
                    title='USD Purchased vs Interest Earned',
                    xaxis_title='Period',
                    yaxis_title='USD Amount',
                    barmode='stack',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with chart_tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # USD Exchange Rate over time
                fig = px.line(df, 
                             x='Period',
                             y='USD Value (₱)',
                             title='USD Exchange Rate Over Time', 
                             labels={"USD Value (₱)": "USD Value (₱)"},
                             markers=True)
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # ROI percentage over time
                fig = px.line(df, 
                             x='Period',
                             y='ROI (%)',
                             title='Return on Investment Over Time', 
                             labels={"ROI (%)": "ROI (%)"},
                             markers=True)
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        with chart_tab3:
            # New chart showing maturity and reinvestment activity
            if 'Maturing Principal ($)' in df.columns and 'Reinvestment ($)' in df.columns:
                fig = go.Figure()
                
                # Add maturing principal bars
                fig.add_trace(go.Bar(
                    x=df['Period'],
                    y=df['Maturing Principal ($)'],
                    name="Maturing Principal",
                    marker_color='#E91E63'
                ))
                
                # Add reinvestment line
                fig.add_trace(go.Scatter(
                    x=df['Period'],
                    y=df['Reinvestment ($)'],
                    name="Reinvestment Amount",
                    line=dict(color='#673AB7', width=3),
                    mode='lines+markers'
                ))
                
                fig.update_layout(
                    title='Maturing Investments and Reinvestment',
                    xaxis_title='Period',
                    yaxis_title='USD Amount',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Data table with formatted values
        time_period = "Monthly" if 'Month' in df.columns else ("Quarterly" if 'Quarter' in df.columns else "Yearly")
        st.subheader(f"{time_period} USD Time Deposit Summary")
        
        display_df = df.copy()
        
        # Format currency values
        display_df['USD Value (₱)'] = display_df['USD Value (₱)'].apply(lambda x: f"₱{x:,.2f}")
        display_df['Total Invested (₱)'] = display_df['Total Invested (₱)'].apply(lambda x: f"₱{x:,.2f}")
        display_df['USD Purchased'] = display_df['USD Purchased'].apply(lambda x: f"${x:,.2f}")
        
        # Format new columns if they exist
        if 'Monthly Investment (₱)' in display_df.columns:
            display_df['Monthly Investment (₱)'] = display_df['Monthly Investment (₱)'].apply(lambda x: f"₱{x:,.2f}")
        if 'Quarterly Investment (₱)' in display_df.columns:
            display_df['Quarterly Investment (₱)'] = display_df['Quarterly Investment (₱)'].apply(lambda x: f"₱{x:,.2f}")
        if 'Maturing Principal ($)' in display_df.columns:
            display_df['Maturing Principal ($)'] = display_df['Maturing Principal ($)'].apply(lambda x: f"${x:,.2f}")
        if 'Reinvestment ($)' in display_df.columns:
            display_df['Reinvestment ($)'] = display_df['Reinvestment ($)'].apply(lambda x: f"${x:,.2f}")
            
        display_df['Interest Earned ($)'] = display_df['Interest Earned ($)'].apply(lambda x: f"${x:,.2f}")
        display_df['Interest Earned (₱)'] = display_df['Interest Earned (₱)'].apply(lambda x: f"₱{x:,.2f}")
        display_df['Total USD Holdings'] = display_df['Total USD Holdings'].apply(lambda x: f"${x:,.2f}")
        display_df['Holdings Value (₱)'] = display_df['Holdings Value (₱)'].apply(lambda x: f"₱{x:,.2f}")
        display_df['ROI (%)'] = display_df['ROI (%)'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button for the data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download USD Time Deposit Data as CSV", 
                          data=csv, 
                          file_name=f'usd_time_deposit_{datetime.now().strftime("%Y%m%d")}.csv',
                          mime='text/csv')
        
        # Explanation of the model
        with st.expander("How this model works"):
            st.markdown("""
            ### USD Time Deposit Investment Model Explanation
            
            This model simulates a monthly investment scheme where each investment matures after 3 months with 2% interest.
            
            **Key assumptions:**
            - Monthly investments are made at the beginning of each month
            - Each investment matures after exactly 3 months with a 2% return
            - When an investment matures, its principal plus interest is reinvested along with new investments
            - USD appreciates against PHP annually by 2% (calculated as initial_rate × (1.02^year))
            - All interest earned is automatically reinvested (compounding effect)
            - Monthly investment amount increases by ₱1,500 each year
            
            **Monthly Process:**
            1. Invest the monthly peso amount (convert to USD at current rate)
            2. Check if any previous investments are maturing this month
            3. Calculate 2% interest on all maturing investments
            4. Reinvest all matured investments (principal + interest)
            5. Track total USD holdings and peso value based on current exchange rate
            
            This approach creates a rolling reinvestment cycle that more accurately models time deposits with fixed maturity periods.
            """)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; border-top: 1px solid #ddd;">
    <p>Financial Modeling Dashboard © 2025</p>
</div>
""", unsafe_allow_html=True)
