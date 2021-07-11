-- MySQL dump 10.13  Distrib 8.0.19, for Win64 (x86_64)
--
-- Host: 127.0.0.1    Database: fitzherbertnikki_db
-- ------------------------------------------------------
-- Server version	8.0.19

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `addresses`
--

DROP TABLE IF EXISTS `addresses`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `addresses` (
  `address_id` int NOT NULL,
  `line_1_number_building` varchar(45) DEFAULT NULL,
  `line_2_number_street` varchar(45) DEFAULT NULL,
  `line_3_area_locality` varchar(45) DEFAULT NULL,
  `city` varchar(45) DEFAULT NULL,
  `zip_postcode` varchar(45) DEFAULT NULL,
  `state_province_county` varchar(45) DEFAULT NULL,
  `country` varchar(45) DEFAULT NULL,
  `other_address_details` varchar(45) DEFAULT NULL,
  PRIMARY KEY (`address_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `addresses`
--

LOCK TABLES `addresses` WRITE;
/*!40000 ALTER TABLE `addresses` DISABLE KEYS */;
/*!40000 ALTER TABLE `addresses` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `patient_addresses`
--

DROP TABLE IF EXISTS `patient_addresses`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `patient_addresses` (
  `date_address_from` datetime NOT NULL,
  `address_id` int NOT NULL,
  `patient_id` int NOT NULL,
  `date_address_to` datetime DEFAULT NULL,
  PRIMARY KEY (`date_address_from`),
  KEY `patient_addresses_to_patients_idx` (`patient_id`),
  KEY `patient_addresses_to_addresses_idx` (`address_id`),
  CONSTRAINT `patient_addresses_to_addresses` FOREIGN KEY (`address_id`) REFERENCES `addresses` (`address_id`),
  CONSTRAINT `patient_addresses_to_patients` FOREIGN KEY (`patient_id`) REFERENCES `patients` (`patient_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `patient_addresses`
--

LOCK TABLES `patient_addresses` WRITE;
/*!40000 ALTER TABLE `patient_addresses` DISABLE KEYS */;
/*!40000 ALTER TABLE `patient_addresses` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `patients`
--

DROP TABLE IF EXISTS `patients`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `patients` (
  `patient_id` int NOT NULL,
  `outpatient_yn` int DEFAULT NULL,
  `hospital_number` int DEFAULT NULL,
  `nhs_number` int DEFAULT NULL,
  `gender` char(1) DEFAULT NULL,
  `date_of_birth` datetime DEFAULT NULL,
  `patient_first_name` varchar(45) DEFAULT NULL,
  `patient_middle_name` varchar(45) DEFAULT NULL,
  `patient_last_name` varchar(45) DEFAULT NULL,
  `height` int DEFAULT NULL,
  `weight` int DEFAULT NULL,
  `next_of_kin` varchar(45) DEFAULT NULL,
  `home_phone` varchar(45) DEFAULT NULL,
  `work_phone` varchar(45) DEFAULT NULL,
  `cell_mobile_phone` varchar(45) DEFAULT NULL,
  `other_patient_details` varchar(45) DEFAULT NULL,
  PRIMARY KEY (`patient_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `patients`
--

LOCK TABLES `patients` WRITE;
/*!40000 ALTER TABLE `patients` DISABLE KEYS */;
/*!40000 ALTER TABLE `patients` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `staff`
--

DROP TABLE IF EXISTS `staff`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `staff` (
  `staff_id` int NOT NULL,
  `staff_category_code` int DEFAULT NULL,
  `gender` char(1) DEFAULT NULL,
  `staff_job_title` varchar(45) DEFAULT NULL,
  `staff_first_name` varchar(45) DEFAULT NULL,
  `staff_middle_name` varchar(45) DEFAULT NULL,
  `staff_last_name` varchar(45) DEFAULT NULL,
  `staff_qualifications` varchar(45) DEFAULT NULL,
  `staff_birth_date` datetime DEFAULT NULL,
  `other_staff_details` varchar(45) DEFAULT NULL,
  PRIMARY KEY (`staff_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `staff`
--

LOCK TABLES `staff` WRITE;
/*!40000 ALTER TABLE `staff` DISABLE KEYS */;
/*!40000 ALTER TABLE `staff` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `staff_addresses`
--

DROP TABLE IF EXISTS `staff_addresses`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `staff_addresses` (
  `date_address_from` datetime NOT NULL,
  `address_id` int NOT NULL,
  `staff_id` int NOT NULL,
  `date_address_to` datetime DEFAULT NULL,
  PRIMARY KEY (`date_address_from`),
  KEY `staff_addresses_to_addresses_idx` (`address_id`),
  KEY `staff_addresses_to_staff_idx` (`staff_id`),
  CONSTRAINT `staff_addresses_to_addresses` FOREIGN KEY (`address_id`) REFERENCES `addresses` (`address_id`),
  CONSTRAINT `staff_addresses_to_staff` FOREIGN KEY (`staff_id`) REFERENCES `staff` (`staff_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `staff_addresses`
--

LOCK TABLES `staff_addresses` WRITE;
/*!40000 ALTER TABLE `staff_addresses` DISABLE KEYS */;
/*!40000 ALTER TABLE `staff_addresses` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-03-19 20:39:39
