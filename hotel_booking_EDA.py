import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import (StandardScaler,MinMaxScaler,
OneHotEncoder,LabelEncoder,RobustScaler)

class HotelBooking:
    def __init__(self,file_path):
        self.df = pd.read_csv(file_path)
        self.cleaned_df = self.df.copy()

        self.num_cols = self.df.select_dtypes(include=['float64','int64']).columns
        self.cat_cols = self.df.select_dtypes(include=['object']).columns

    # 1. DATASET SUMMARY
    def dataset_summary(self):
        print("DATASET SUMMARY")
        print("\nShape :", self.df.shape)

        print("\n First 10 Rows")
        print(self.df.head(10))
        
        print("\nLast 10 Rows")
        print(self.df.tail(10))

        print("\nColumns")
        print(self.df.columns)

        print("\nNumerical Columns")
        print(self.num_cols)

        print("\nCategorical Columns")
        print(self.cat_cols)

        print("\nInfo")
        print(self.df.info())

        print("\nDescribe")
        print(self.df.describe())

        print("\nMemory Usage")
        print(self.df.memory_usage(deep=True))

    # 2. MISSING VALUES
    def missing_value_report(self):
        missing=pd.DataFrame({"Missing Values":self.df.isnull().sum(),
                     "Percentage":round(self.df.isnull().sum() / len(self.df) * 100, 2)
        })
        missing=missing[missing['Missing Values']>0].sort_values(by='Percentage',ascending=False)
        

        plt.figure(figsize=(12, 6))
        sns.heatmap(self.df.isnull(), cbar=False)
        plt.title('Missing Value Heatmap')
        plt.show()

        return missing
    
    # 3. DUPLICATES
    def duplicate_report(self):
        duplicates = self.df.duplicated().sum()
        print("Duplicate Rows :", duplicates)
        return duplicates
    
    def remove_duplicates(self):
        before = self.cleaned_df.shape[0]
        self.cleaned_df.drop_duplicates(inplace=True)
        after = self.cleaned_df.shape[0]
        print(f"{before-after} duplicate rows removed.")

    # 4. UNIQUE VALUE ANALYSIS
    def unique_value_report(self):
        report = []
        for col in self.df.columns:
            unique = self.df[col].nunique()
            mode = self.df[col].mode()[0]
            freq = (self.df[col].value_counts(normalize=True).iloc[0]* 100)
            # The normalize=True parameter in value_counts() converts the 
            # counts into proportions (percentages) instead of returning the actual counts.
            report.append([col, unique, mode,round(freq, 2)])

        report = pd.DataFrame(report,columns=['Column','Unique Values','Most Frequent','Frequency %'])
        return report
    
    # 5. MISSING VALUE IMPUTATION
    def impute_missing_values(self):
        # For Numerical Columns
        for col in self.num_cols:
            # Check for outliers using IQR
            q1 = self.cleaned_df[col].quantile(0.25)
            q3 = self.cleaned_df[col].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            has_outliers = ((self.cleaned_df[col] < lower) | (self.cleaned_df[col] > upper)).any()
        
            # Imputation
            if has_outliers:
                self.cleaned_df[col] = (self.cleaned_df[col].fillna(
                    self.cleaned_df[col].median()
                    )
                )
            else:
                self.cleaned_df[col] = (self.cleaned_df[col].fillna(
                        self.cleaned_df[col].mean()
                    )
                )

        # Categorical columns
        for col in self.cat_cols:
            self.cleaned_df[col] = (self.cleaned_df[col].fillna(
                    self.cleaned_df[col].mode()[0]
                )
            )
        print("Missing values imputed successfully.")
        
    # 6. UNIVARIATE ANALYSIS
    def numerical_summary(self):
        summary = pd.DataFrame()

        summary['Mean'] = (
            self.cleaned_df[self.num_cols].mean()
        )

        summary['Median'] = (
            self.cleaned_df[self.num_cols].median()
        )

        summary['Std'] = (
            self.cleaned_df[self.num_cols].std()
        )

        summary['Variance'] = (
            self.cleaned_df[self.num_cols].var()
        )

        summary['Min'] = (
            self.cleaned_df[self.num_cols].min()
        )

        summary['Max'] = (
            self.cleaned_df[self.num_cols].max()
        )

        summary['Skewness'] = (
            self.cleaned_df[self.num_cols].skew()
        )

        return summary      
        
    # 7. PLOT ALL NUMERICAL COLUMNS
    
    def plot_numerical_columns(self):

        for col in self.num_cols:
            if self.cleaned_df[col].nunique() <= 1:
                continue

            # Histogram with KDE
            plt.figure(figsize=(8, 4))
            sns.histplot(self.cleaned_df[col],kde=True)
            plt.title(f'Histogram : {col}')
            plt.show()

            # Boxplot
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=self.cleaned_df[col])
            plt.title(f'Boxplot : {col}')
            plt.show()
    
    # 8. CATEGORICAL ANALYSIS
    
    def categorical_summary(self):
        for col in self.cat_cols:
            print("\n", "=" * 50)
            print(col)
            print("=" * 50)

            summary = pd.DataFrame({
            'Count': self.cleaned_df[col].value_counts(),
            'Percentage': round(self.cleaned_df[col].value_counts(normalize=True) * 100, 2)
        })
            print(summary)
            ## Count Plot
            plt.figure(figsize=(8, 4))
            plt.title(f'Countplot : {col}')
            sns.countplot(
                data=self.cleaned_df,
                x=col
            )
            plt.xticks(rotation=90)
            plt.show()
            
            ## Pie Chart
            counts = self.cleaned_df[col].value_counts()

            plt.figure(figsize=(8, 8))
            plt.pie(
                counts,
                labels=counts.index,
                autopct='%1.1f%%'
            )
            plt.title(f'Pie Chart : {col}')
            plt.show()

    # 9. IQR OUTLIERS
    def iqr_outliers(self):

        report = []

        for col in self.num_cols:

            q1 = self.cleaned_df[col].quantile(.25)
            q3 = self.cleaned_df[col].quantile(.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = self.cleaned_df[
                (self.cleaned_df[col] < lower)
                |
                (self.cleaned_df[col] > upper)
            ]

            report.append(
                [col, len(outliers)]
            )
        
        return pd.DataFrame(
            report,
            columns=['Column', 'Outliers']
        )
    
    # 10. Z SCORE OUTLIERS
    def zscore_outliers(self):

        report = []

        for col in self.num_cols:

            z = (
                self.cleaned_df[col]
                - self.cleaned_df[col].mean()
            ) / self.cleaned_df[col].std()

            count = np.sum(np.abs(z) > 3)

            report.append([col, count])

        return pd.DataFrame(
            report,
            columns=['Column', 'Outliers']
        )


    ## Remove outliers
     
    def remove_outliers(self):

        for col in self.num_cols:

            q1 = self.cleaned_df[col].quantile(0.25)
            q3 = self.cleaned_df[col].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            self.cleaned_df = self.cleaned_df[
                (self.cleaned_df[col] >= lower)
                &
                (self.cleaned_df[col] <= upper)
            ]

        print('Outliers removed successfully.')

    ## Capping 
    def cap_outliers(self):

        for col in self.num_cols:

            q1 = self.cleaned_df[col].quantile(0.25)
            q3 = self.cleaned_df[col].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            self.cleaned_df[col] = self.cleaned_df[col].clip(lower, upper)
        print('Outliers capped successfully.')
    
    ## Winsorization
    def winsorize_outliers(self):

        for col in self.num_cols:

            lower = (self.cleaned_df[col].quantile(0.05))
            upper = (self.cleaned_df[col].quantile(0.95))
            self.cleaned_df[col] = self.cleaned_df[col].clip(lower, upper)

        print('Winsorization completed successfully.')

        
    # 11. FEATURE ENGINEERING
   
    def feature_engineering(self):

        df = self.cleaned_df

        df['total_guests'] = ( df['adults'] + df['children'] + df['babies'])

        df['stay_duration'] = (df['stays_in_weekend_nights'] + df['stays_in_week_nights'])

        df['estimated_revenue'] = (df['adr'] * df['stay_duration'])

        df['high_value_customer'] = np.where(
            df['adr'] >
            df['adr'].quantile(.75),
            1,
            0
        )

        df['guest_type'] = np.where(
            df['total_guests'] == 1,
            'Solo',
            np.where(
                df['total_guests'] == 2,
                'Couple',
                'Family'
            )
        )

        print("Feature Engineering Completed.")

    # 12. Bivariate Analysis
    # Numerical vs Numerical
    def correlation_analysis(self):

        corr = (self.cleaned_df[self.num_cols].corr())

        plt.figure(figsize=(14, 10))

        sns.heatmap(
            corr,
            annot=True,
            cmap='coolwarm'
        )

        plt.show()

    

    # 13. SCALING
    def scaling(self):

        scaler1 = StandardScaler()
        scaler2 = MinMaxScaler()
        scaler3 = RobustScaler()

        standard = scaler1.fit_transform(
            self.cleaned_df[self.num_cols]
        )

        minmax = scaler2.fit_transform(
            self.cleaned_df[self.num_cols]
        )

        robust = scaler3.fit_transform(
            self.cleaned_df[self.num_cols]
        )

        print("Scaling Completed.")

        return standard, minmax, robust

    # 14. Hotel Business Analysis
    def hotel_business_analysis(self):

        ## Cancellation Analysis
        ## CountPlot
        plt.figure(figsize=(6, 4))
        sns.countplot(
            data=self.cleaned_df,
            x='is_canceled'
        )

        plt.title('Cancellation Count')
        plt.show()

        # 2. Pie Chart
        counts = (self.cleaned_df['is_canceled'].value_counts())
        plt.figure(figsize=(6, 6))
        plt.pie(counts,labels=['Not Cancelled', 'Cancelled'],autopct='%1.1f%%')
        plt.title('Cancellation Percentage')
        plt.show()

        print("1. What is the cancellation rate?")
        cancellation_rate = (self.cleaned_df['is_canceled'].mean()* 100)
        print('Cancellation Rate:',round(cancellation_rate, 2),'%')
        
        print("2.What percentage of bookings are successfully completed?")
        successful_rate = (
        (1 - self.cleaned_df['is_canceled'].mean())* 100)
        print('Successful Bookings:',round(successful_rate, 2),'%')  

        print("3.How much potential revenue is lost due to cancellations?")
        self.cleaned_df['estimated_revenue'] = (self.cleaned_df['adr'] * self.cleaned_df['stay_duration'])
        lost_revenue = (self.cleaned_df.loc[self.cleaned_df['is_canceled'] == 1,
        'estimated_revenue'].sum())
        print('Potential Revenue Lost:',round(lost_revenue, 2))

        print("4.Which hotel experiences higher cancellations?")
        hotel_cancel = (self.cleaned_df.groupby('hotel')['is_canceled'].mean()* 100)
        print(hotel_cancel)
        hotel_cancel.plot(kind='bar',figsize=(6, 4))
        plt.ylabel('Cancellation %')
        plt.title('Hotel-wise Cancellation Rate')
        plt.show()

    ## 15.Seasonal Booking Analysis
    def seasonal_booking_analysis(self):

        ## Monthly booking trend
        month_order = ['January','February','March','April','May','June','July','August',
                        'September','October','November','December']

        monthly = (
        self.cleaned_df['arrival_date_month'].value_counts().reindex(month_order))

        monthly.plot(
            kind='line',
            marker='o',
            figsize=(10, 5)
        )

        plt.title('Monthly Booking Trend')
        plt.ylabel('Bookings')
        plt.show()

        ## Year-wise booking trend
        yearly = (self.cleaned_df['arrival_date_year'].value_counts().sort_index())

        yearly.plot(kind='bar',figsize=(8, 4))

        plt.title('Year-wise Booking Trend')
        plt.ylabel('Bookings')
        plt.show()

        ## Month-Year heatmap
        heatmap_data = pd.crosstab(self.cleaned_df['arrival_date_month'],self.cleaned_df['arrival_date_year'])
        heatmap_data = (heatmap_data.reindex(month_order))
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data,annot=True)
        plt.show()
        
        print("1. Which month receives maximum bookings?")
        print(monthly.idxmax())

        print("2. Which month receives minimum bookings?")
        print(monthly.idxmin())

        print("3. Is there seasonality in hotel demand?")
        print("YES")

        print("4. During which season should staffing be increased?")
        print("Increase staffing during ",monthly.idxmax()," and nearby peak months.")
    
    # 16. REVENUE ANALYSIS
    def revenue_analysis(self):

        revenue = (self.cleaned_df.groupby('hotel')['estimated_revenue'].sum()
            .sort_values(ascending=False)
        )

        print(revenue)

        revenue.plot(
            kind='bar',
            figsize=(8, 5)
        )

        plt.title('Revenue By Hotel')
        plt.show()

        print("1. Which hotel generates the highest revenue?")
        print(self.cleaned_df.groupby('hotel')['estimated_revenue'].sum().idxmax())

        print("2. Which month generates the highest revenue?")
        print(self.cleaned_df.groupby('arrival_date_month')['estimated_revenue'].sum().idxmax())

        print("3. Which customer segment generates the most revenue?")
        print(self.cleaned_df.groupby('market_segment')['estimated_revenue'].sum().idxmax())

        print("4. Which country contributes the most revenue?")
        print(self.cleaned_df.groupby('country')['estimated_revenue'].sum().idxmax())

    ##17.Cancellation Risk Analysis
    def cancellation_risk(self):
        print("1. Which customer type cancels most?")
        print(self.cleaned_df.groupby('customer_type')['is_canceled'].mean().sort_values(ascending=False))
        
        print("2. Which market segment cancels most?")
        print(self.cleaned_df.groupby('market_segment')['is_canceled'].mean().sort_values(ascending=False))

        print("3. Which booking channel has highest cancellation?")
        print(self.cleaned_df.groupby('distribution_channel')['is_canceled'].mean().sort_values(ascending=False))

        print("4. Which deposit type has highest cancellation?")
        print(self.cleaned_df.groupby('deposit_type')['is_canceled'].mean().sort_values(ascending=False))


    ## 18.Customer Behavior Analysis
    def customer_behavior_analysis(self):
        print("1. Who books furthest in advance?")
        print(self.cleaned_df.groupby('customer_type')['lead_time'].mean().sort_values(ascending=False))

        print("2. Who stays longest?")
        print(self.cleaned_df.groupby('customer_type')['stay_duration'].mean().sort_values(ascending=False))

        print("3. Which guests make special requests most often?")
        print(self.cleaned_df.groupby('customer_type')['total_of_special_requests'].mean().sort_values(ascending=False))

        print("4. Which guest group is most profitable?")
        print(self.cleaned_df.groupby('customer_type')['estimated_revenue'].sum().sort_values(ascending=False))

    # 19. ML READINESS REPORT

    def ml_readiness_report(self):

        print("=" * 60)
        print("ML READINESS REPORT")
        print("=" * 60)

        print("\nColumns To Encode")
        print(self.cat_cols)

        print("\nColumns To Scale")
        print(self.num_cols)

        print("\nMissing Values")
        print(self.cleaned_df.isnull().sum().sum())

        print("\nDuplicate Rows")
        print(self.cleaned_df.duplicated().sum())

    # RUN EVERYTHING AUTOMATICALLY
        
    def run_complete_analysis(self):

        self.dataset_summary()

        print("\nMissing Values")
        print(
            self.missing_value_report()
        )

        print("\nDuplicate Rows")
        self.duplicate_report()

        self.remove_duplicates()

        print(
            self.unique_value_report()
        )

        self.impute_missing_values()

        print(
            self.numerical_summary()
        )

        self.plot_numerical_columns()

        self.categorical_summary()

        print(
            self.iqr_outliers()
        )

        print(
            self.zscore_outliers()
        )
        self.remove_outliers()

        self.cap_outliers()

        self.winsorize_outliers()

        self.feature_engineering()

        self.correlation_analysis()

        self.scaling()

        self.hotel_business_analysis()

        self.seasonal_booking_analysis()

        self.revenue_analysis()

        self.cancellation_risk()

        self.customer_behavior_analysis()

        self.ml_readiness_report()

        print(
            "\nALL ANALYSIS COMPLETED SUCCESSFULLY."
        )

# HOW TO RUN
obj = HotelBooking(
    r"C:\Users\jhala\OneDrive\Desktop\hotel_bookings_extracted.csv"
)

obj.run_complete_analysis()