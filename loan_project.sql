-- Select the database
USE loan_project;

-- 1. Show all tables
SHOW TABLES;

-- 2. Preview data (Limited to 10 rows so it doesn't freeze)
SELECT * FROM loan_project.loan_data LIMIT 10;

-- 3. Total Applications
SELECT COUNT(*) AS total_applications
FROM loan_project.loan_data;

-- 4. Average Income
SELECT AVG(person_income) AS avg_income 
FROM loan_project.loan_data;

-- 5. Average Credit Score
SELECT AVG(credit_score) AS avg_credit_score
FROM loan_project.loan_data;

-- 6. Defaulter Rate
SELECT
  ROUND(
    100.0 * SUM(loan_status) / COUNT(*),
    2
  ) AS default_rate
FROM loan_project.loan_data;

-- 7. High Risk Customers 
-- Uses loan_percent_income (e.g., 0.40 means 40% of income goes to loan)
SELECT *
FROM loan_project.loan_data
WHERE credit_score < 600
  AND loan_percent_income > 0.40;

-- 8. Income Group Analysis
SELECT
  CASE
    WHEN person_income < 30000 THEN 'Low'
    WHEN person_income < 70000 THEN 'Medium'
    ELSE 'High'
  END AS income_group,
  COUNT(*) AS applicants
FROM loan_project.loan_data
GROUP BY income_group;

-- 9. Default Rate by Income Group
SELECT
  CASE
    WHEN person_income < 30000 THEN 'Low'
    WHEN person_income < 70000 THEN 'Medium'
    ELSE 'High'
  END AS income_group,
  ROUND(
    100.0 * SUM(loan_status) / COUNT(*),
    2
  ) AS default_rate
FROM loan_project.loan_data
GROUP BY income_group;

-- 10. Top 10 Highest Loan Amounts
SELECT *
FROM loan_project.loan_data
ORDER BY loan_amnt DESC
LIMIT 10;
#CTE
WITH risk_customers AS
(
SELECT *
FROM loan_project.loan_data
WHERE credit_score < 600
)

SELECT COUNT(*)
FROM risk_customers;
#Window Function
SELECT
person_income,

RANK() OVER(
ORDER BY person_income DESC
) AS income_rank

FROM loan_project.loan_data;
#Running Total

