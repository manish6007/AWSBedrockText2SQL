
  
  
CREATE EXTERNAL TABLE `customers`(
  `CustomerID` int COMMENT 'from deserializer', 
  `FirstName` string COMMENT 'from deserializer', 
  `LastName` string COMMENT 'from deserializer', 
  `Email` string COMMENT 'from deserializer', 
  `PhoneNumber` string COMMENT 'from deserializer')
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.serde2.OpenCSVSerde' 
WITH SERDEPROPERTIES ( 
  'escapeChar'='\\', 
  'quoteChar'='\"', 
  'separatorChar'=',') 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.mapred.TextInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION
  's3://athena-datasource-cg/customers1'
TBLPROPERTIES (
  'skip.header.line.count'='1', 
  'transient_lastDdlTime'='1720106395');
  
CREATE EXTERNAL TABLE `orders`(
  `OrderID` int COMMENT 'from deserializer', 
  `CustomerID` int COMMENT 'from deserializer', 
  `OrderDate` date COMMENT 'from deserializer', 
  `OrderAmount` double COMMENT 'from deserializer', 
  `OrderStatus` string COMMENT 'from deserializer')
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.serde2.OpenCSVSerde' 
WITH SERDEPROPERTIES ( 
  'escapeChar'='\\', 
  'quoteChar'='\"', 
  'separatorChar'=',') 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.mapred.TextInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION
  's3://athena-datasource-cg/orders'
TBLPROPERTIES (
  'skip.header.line.count'='1', 
  'transient_lastDdlTime'='1720106395');
  
CREATE EXTERNAL TABLE `products`(
  `ProductID` int COMMENT 'from deserializer', 
  `ProductName` string COMMENT 'from deserializer', 
  `Category` string COMMENT 'from deserializer', 
  `Price` double COMMENT 'from deserializer', 
  `StockQuantity` int COMMENT 'from deserializer')
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.serde2.OpenCSVSerde' 
WITH SERDEPROPERTIES ( 
  'escapeChar'='\\', 
  'quoteChar'='\"', 
  'separatorChar'=',') 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.mapred.TextInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION
  's3://athena-datasource-cg/products'
TBLPROPERTIES (
  'skip.header.line.count'='1', 
  'transient_lastDdlTime'='1720106395');
  
CREATE EXTERNAL TABLE `order_items`(
  `OrderItemID` int COMMENT 'from deserializer', 
  `OrderID` int COMMENT 'from deserializer', 
  `ProductID` int COMMENT 'from deserializer', 
  `Quantity` int COMMENT 'from deserializer', 
  `UnitPrice` double COMMENT 'from deserializer')
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.serde2.OpenCSVSerde' 
WITH SERDEPROPERTIES ( 
  'escapeChar'='\\', 
  'quoteChar'='\"', 
  'separatorChar'=',') 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.mapred.TextInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION
  's3://athena-datasource-cg/order_items'
TBLPROPERTIES (
  'skip.header.line.count'='1', 
  'transient_lastDdlTime'='1720106395');
  

  
