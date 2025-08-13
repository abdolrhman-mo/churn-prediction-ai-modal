Phase 2: data assessment
- [X] check data completness
- [X] check for duplicates
- [X] check for wrong data types

---

Phase 5: Data Preprocessing
- A Remove useless columns
  - [X] Drop customerID column (just a random number, doesn't help predict churn)
- B Convert text to numbers (encoding)
  - [ ] Convert Churn column: "Yes" → 1, "No" → 0
  - [ ] Convert Gender column: "Male" → 1, "Female" → 0
  - [ ] Convert Contract column: create separate columns for each contract type
  - [ ] Convert other categorical columns to numeric format
- C Prepare data for training
  - [ ] Separate features (X) from target (y)
  - [ ] Split data: 80% training, 20% testing
  - [ ] Apply one-hot encoding to remaining categorical variables
  - [ ] Scale numerical features if needed
- D Verify data shape and types
  - [ ] Check final DataFrame shape
  - [ ] Confirm all columns are numeric
  - [ ] Verify no missing values remain