# Churn Prediction AI Model

## Telecom Customer Churn Dataset - Data Dictionary

### Customer Demographics
- **customerID**: Unique customer identifier
- **gender**: Customer gender (Male/Female)
- **SeniorCitizen**: Whether customer is 65+ years old (0=No, 1=Yes)
- **Partner**: Has partner/spouse (Yes/No)
- **Dependents**: Has dependents like children (Yes/No)

### Account Information  
- **tenure**: Number of months with company
- **Contract**: Contract type (Month-to-month, One year, Two year)
- **PaperlessBilling**: Uses electronic billing (Yes/No)
- **PaymentMethod**: Payment method (Electronic check, Mailed check, Bank transfer, Credit card)

### Services
- **PhoneService**: Has phone service (Yes/No)
- **MultipleLines**: Multiple phone lines (Yes/No/No phone service)
- **InternetService**: Internet type (DSL, Fiber optic, No)

### Add-on Services
- **OnlineSecurity**: Internet security add-on (Yes/No/No internet service)
- **OnlineBackup**: Online backup service (Yes/No/No internet service)
- **DeviceProtection**: Device protection plan (Yes/No/No internet service)
- **TechSupport**: Technical support service (Yes/No/No internet service)
- **StreamingTV**: TV streaming service (Yes/No/No internet service)
- **StreamingMovies**: Movie streaming service (Yes/No/No internet service)

### Financial & Target
- **MonthlyCharges**: Current monthly charges ($)
- **TotalCharges**: Total charges over customer lifetime ($)
- **Churn**: **TARGET VARIABLE** - Customer left company (Yes/No)