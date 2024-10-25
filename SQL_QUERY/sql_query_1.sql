
--Create Table
CREATE TABLE Dog_Breed_Classifier (
    dog_image VARBINARY(MAX),
    dog_predicted_name TEXT,
    gemini_response TEXT
);


--Select rows
SELECT TOP (1000) [dog_image]
      ,[dog_predicted_name]
      ,[gemini_response]
  FROM [dog_prediction].[dbo].[main_data]




--View Table format for data insertion
SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'main_data';


