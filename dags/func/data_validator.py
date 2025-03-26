class Validator:
    def __init__(self, df):
        self.df = df
        self.report = {}

    def schema_validation(self):
        expected_schema = {
            'PassengerId': 'int64', 'Survived': 'int64', 'Pclass': 'int64',
            'Name': 'object', 'Sex': 'object', 'Age': 'float64',
            'SibSp': 'int64', 'Parch': 'int64', 'Ticket': 'object',
            'Fare': 'float64', 'Cabin': 'object', 'Embarked': 'object'
            }
        
        schema_validation = {

            col: (self.df[col].dtypes , expected_schema.get(col, 'unknown')) for col in self.df.columns if self.df[col].dtypes != expected_schema.get(col, self.df[col].dtypes)

        }
        
        # Update the report
        self.report['Schema Validation'] = schema_validation if schema_validation else "No Issue with Schema"


    def duplicate_check(self):
        """Check for duplicate rows and IDs"""
    
        duplicates = self.df.duplicated().sum()
        dup_ids = self.df['PassengerId'].duplicated().sum()
        self.report['Duplicate Rows'] = duplicates
        self.report['Duplicate IDs'] = dup_ids

    def missing_value(self):
        missing_values = {}

        # Calculate missing value percentage for each column
        for col in self.df.columns:
            missing_percentage = (self.df[col].isnull().sum() / len(self.df)) * 100
            if missing_percentage > 0:
                missing_values[col] = f"{missing_percentage:.2f}%"

        # Update the report
        self.report['Missing Values'] = (
            missing_values if missing_values else "No missing values"
        )

    def validate(self):
        self.schema_validation()
        self.missing_value()
        self.duplicate_check()
        return self.report