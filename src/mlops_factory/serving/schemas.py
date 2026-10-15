"""Pydantic v2 schemas and request/response contracts for the inference API."""

from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """
    Schema for customer demographic, service, and account attributes.
    Supports both natural dataset column names and clean snake_case aliases.
    """
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    # Demographics
    gender: str = Field("Female", alias="Gender", description="Gender of customer ('Male' or 'Female')")
    age: int = Field(42, ge=0, le=120, alias="Age", description="Customer age in years")
    under_30: str = Field("No", alias="Under 30", description="Whether customer is under 30 ('Yes' or 'No')")
    senior_citizen: str = Field("No", alias="Senior Citizen", description="Whether customer is a senior citizen ('Yes' or 'No')")
    married: str = Field("Yes", alias="Married", description="Marital status ('Yes' or 'No')")
    dependents: str = Field("No", alias="Dependents", description="Whether customer has dependents ('Yes' or 'No')")
    number_of_dependents: int = Field(0, ge=0, le=30, alias="Number of Dependents")
    population: int = Field(25000, ge=0, alias="Population", description="Population of customer's postal area")

    # Referrals & Relationship
    referred_a_friend: str = Field("No", alias="Referred a Friend")
    number_of_referrals: int = Field(0, ge=0, le=50, alias="Number of Referrals")
    tenure_in_months: int = Field(24, ge=0, le=120, alias="Tenure in Months")
    offer: str = Field("None", alias="Offer", description="Marketing offer accepted ('Offer A'-'Offer E', or 'None')")

    # Services
    phone_service: str = Field("Yes", alias="Phone Service")
    avg_monthly_long_distance_charges: float = Field(15.0, ge=0.0, alias="Avg Monthly Long Distance Charges")
    multiple_lines: str = Field("No", alias="Multiple Lines")
    internet_service: str = Field("Yes", alias="Internet Service")
    internet_type: str = Field("Fiber Optic", alias="Internet Type", description="'DSL', 'Fiber Optic', 'Cable', or 'None'")
    avg_monthly_gb_download: int = Field(25, ge=0, alias="Avg Monthly GB Download")
    online_security: str = Field("No", alias="Online Security")
    online_backup: str = Field("Yes", alias="Online Backup")
    device_protection_plan: str = Field("No", alias="Device Protection Plan")
    premium_tech_support: str = Field("No", alias="Premium Tech Support")
    streaming_tv: str = Field("Yes", alias="Streaming TV")
    streaming_movies: str = Field("Yes", alias="Streaming Movies")
    streaming_music: str = Field("No", alias="Streaming Music")
    unlimited_data: str = Field("Yes", alias="Unlimited Data")

    # Billing & Financials
    contract: str = Field("Month-to-Month", alias="Contract", description="'Month-to-Month', 'One Year', 'Two Year'")
    paperless_billing: str = Field("Yes", alias="Paperless Billing")
    payment_method: str = Field("Bank Withdrawal", alias="Payment Method", description="'Bank Withdrawal', 'Credit Card', 'Mailed Check'")
    monthly_charge: float = Field(75.50, ge=0.0, alias="Monthly Charge")
    total_charges: float = Field(1812.0, ge=0.0, alias="Total Charges")
    total_refunds: float = Field(0.0, ge=0.0, alias="Total Refunds")
    total_extra_data_charges: int = Field(0, ge=0, alias="Total Extra Data Charges")
    total_long_distance_charges: float = Field(360.0, ge=0.0, alias="Total Long Distance Charges")
    total_revenue: float = Field(2172.0, ge=0.0, alias="Total Revenue")
    cltv: int = Field(4500, ge=0, alias="CLTV", description="Customer Lifetime Value index")

    def to_dataframe_row(self) -> Dict[str, Any]:
        """Converts model fields to natural dataset column naming matching training schema."""
        return {
            "Gender": self.gender,
            "Age": self.age,
            "Under 30": self.under_30,
            "Senior Citizen": self.senior_citizen,
            "Married": self.married,
            "Dependents": self.dependents,
            "Number of Dependents": self.number_of_dependents,
            "Population": self.population,
            "Referred a Friend": self.referred_a_friend,
            "Number of Referrals": self.number_of_referrals,
            "Tenure in Months": self.tenure_in_months,
            "Offer": self.offer,
            "Phone Service": self.phone_service,
            "Avg Monthly Long Distance Charges": self.avg_monthly_long_distance_charges,
            "Multiple Lines": self.multiple_lines,
            "Internet Service": self.internet_service,
            "Internet Type": self.internet_type,
            "Avg Monthly GB Download": self.avg_monthly_gb_download,
            "Online Security": self.online_security,
            "Online Backup": self.online_backup,
            "Device Protection Plan": self.device_protection_plan,
            "Premium Tech Support": self.premium_tech_support,
            "Streaming TV": self.streaming_tv,
            "Streaming Movies": self.streaming_movies,
            "Streaming Music": self.streaming_music,
            "Unlimited Data": self.unlimited_data,
            "Contract": self.contract,
            "Paperless Billing": self.paperless_billing,
            "Payment Method": self.payment_method,
            "Monthly Charge": self.monthly_charge,
            "Total Charges": self.total_charges,
            "Total Refunds": self.total_refunds,
            "Total Extra Data Charges": self.total_extra_data_charges,
            "Total Long Distance Charges": self.total_long_distance_charges,
            "Total Revenue": self.total_revenue,
            "CLTV": self.cltv,
        }


class PredictionRequest(BaseModel):
    """Wrapper for a single customer churn prediction request."""
    customer: CustomerFeatures


class PredictionResponse(BaseModel):
    """Prediction output schema."""
    churn_prediction: bool = Field(..., description="True if customer is predicted to churn, False otherwise")
    churn_probability: float = Field(..., description="Estimated probability of churning (0.0 to 1.0)")
    risk_tier: str = Field(..., description="Categorical risk rating: 'Low', 'Medium', or 'High'")
    recommended_action: str = Field(..., description="Operational retention action guidance")
    model_name: str
    model_version: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    """Batch prediction request with maximum item threshold."""
    customers: List[CustomerFeatures] = Field(..., max_length=1000, description="List of customer profiles (up to 1,000)")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response containing list of predictions and aggregate summary."""
    total_processed: int
    predicted_churners_count: int
    predictions: List[PredictionResponse]
    batch_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    uptime_seconds: float


class MetadataResponse(BaseModel):
    model_name: str
    trained_at_utc: str
    model_sha256: str
    total_features: int
    scikit_learn_version: str
    quality_gate_passed: bool
    test_metrics: Any
