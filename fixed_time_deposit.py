import pandas as pd

def simulate_usd_time_deposit_monthly(monthly_peso_investment, initial_usd_value, investment_years, show_monthly=False):
    """
    Simulates a USD time deposit investment scheme with monthly investments and proper compounding.
    
    Args:
        monthly_peso_investment: Initial monthly investment in PHP
        initial_usd_value: Starting USD-PHP exchange rate
        investment_years: Total investment period in years
        show_monthly: If True, returns monthly data; if False, returns yearly data
        
    Returns:
        DataFrame with investment results over time
    """
    # Constants
    MATURITY_INTEREST_RATE = 0.02  # 2% interest at maturity (after 3 months)
    ANNUAL_USD_APPRECIATION = 0.025  # 2.0% annual USD value increase
    ANNUAL_INVESTMENT_INCREMENT = 5000  # Increase monthly investment by ₱1,500 each year
    MATURITY_MONTHS = 3  # Each investment matures after 3 months
    
    # Initialize tracking variables
    usd_value = initial_usd_value
    total_usd_holdings = 0
    total_peso_invested = 0
    current_monthly_investment = monthly_peso_investment
    
    # Store data for all months
    monthly_data = []
    
    # Create a list to track investments that will mature each month
    # Each entry is (principal_usd, month_to_mature)
    maturing_investments = []
    
    # Calculate total number of months
    total_months = investment_years * 12
    
    # Process month by month
    for month in range(1, total_months + 1):
        current_year = (month - 1) // 12 + 1
        current_month = ((month - 1) % 12) + 1
        
        # Calculate USD exchange rate for this period (updates annually)
        if month % 12 == 1 and month > 1:  # First month of a new year (except first year)
            usd_value = initial_usd_value * (1 + ANNUAL_USD_APPRECIATION) ** (current_year - 1)
            # Increase monthly investment amount annually
            current_monthly_investment = monthly_peso_investment + ANNUAL_INVESTMENT_INCREMENT * (current_year - 1)
        
        # Calculate this month's new investment in USD
        monthly_peso_amt = current_monthly_investment
        monthly_usd_investment = monthly_peso_amt / usd_value
        
        # Add to total peso invested
        total_peso_invested += monthly_peso_amt
        
        # Track maturity for this month's investment
        maturity_month = month + MATURITY_MONTHS
        maturing_investments.append((monthly_usd_investment, maturity_month))
        
        # Process maturing investments for this month
        maturing_principal = 0
        interest_earned = 0
        
        # Find investments that mature this month and calculate interest
        remaining_investments = []
        for principal, maturity in maturing_investments:
            if maturity == month:
                # This investment is maturing this month
                maturing_principal += principal
                interest_earned += principal * MATURITY_INTEREST_RATE
            else:
                # This investment hasn't matured yet
                remaining_investments.append((principal, maturity))
        
        # Replace our list with only non-matured investments
        maturing_investments = remaining_investments
        
        # Reinvest matured amounts (principal + interest)
        reinvestment = maturing_principal + interest_earned
        
        # If there's any matured investment, add it back to maturing_investments with new maturity date
        if reinvestment > 0:
            maturing_investments.append((reinvestment, month + MATURITY_MONTHS))
        
        # Update total holdings - now only adding new investment and interest (principal is already counted)
        total_usd_holdings = total_usd_holdings + monthly_usd_investment + interest_earned  
        
        # Calculate peso values based on current exchange rate
        peso_value_end_of_month = total_usd_holdings * usd_value
        
        # Calculate ROI
        roi = ((peso_value_end_of_month / total_peso_invested) - 1) * 100 if total_peso_invested > 0 else 0
        
        # Store data for this month
        monthly_data.append({
            "Year": current_year,
            "Month": current_month,
            "Overall Month": month,
            "Period": f"Y{current_year}M{current_month}",
            "USD Value (₱)": usd_value,
            "Monthly Investment (₱)": monthly_peso_amt,
            "Total Invested (₱)": total_peso_invested,
            "USD Purchased": monthly_usd_investment,
            "Maturing Principal ($)": maturing_principal,
            "Interest Earned ($)": interest_earned,
            "Interest Earned (₱)": interest_earned * usd_value,
            "Reinvestment ($)": reinvestment,
            "Total USD Holdings": total_usd_holdings,
            "Holdings Value (₱)": peso_value_end_of_month,
            "ROI (%)": roi
        })
    
    # Create monthly DataFrame
    monthly_df = pd.DataFrame(monthly_data)
    
    # If show_monthly is True, return monthly data
    if show_monthly:
        return monthly_df
    
    # Otherwise, aggregate by year for a cleaner view (instead of quarterly)
    yearly_data = []
    for year in range(1, investment_years + 1):
        # Find months in this year
        year_months = monthly_df[monthly_df["Year"] == year]
        
        if not year_months.empty:
            # Get data for last month in year
            last_month = year_months.iloc[-1]
            
            # Sum up values for the year
            yearly_investment = year_months["Monthly Investment (₱)"].sum()
            yearly_usd_purchased = year_months["USD Purchased"].sum()
            yearly_interest = year_months["Interest Earned ($)"].sum()
            yearly_maturing = year_months["Maturing Principal ($)"].sum()
            yearly_reinvestment = year_months["Reinvestment ($)"].sum()
            
            yearly_data.append({
                "Year": year,
                "Period": f"Age {year + 30}",
                "USD Value (₱)": last_month["USD Value (₱)"],
                "Monthly Investment (₱)": yearly_investment / 12,
                "Total Invested (₱)": last_month["Total Invested (₱)"],
                "USD Purchased": yearly_usd_purchased,
                "Maturing Principal ($)": yearly_maturing,
                "Interest Earned ($)": yearly_interest,
                "Interest Earned (₱)": yearly_interest * last_month["USD Value (₱)"],
                "Reinvestment ($)": yearly_reinvestment,
                "Total USD Holdings": last_month["Total USD Holdings"],
                "Holdings Value (₱)": last_month["Holdings Value (₱)"],
                "ROI (%)": last_month["ROI (%)"]
            })
    
    # Return yearly aggregated data
    return pd.DataFrame(yearly_data)
