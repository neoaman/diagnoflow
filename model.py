from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

import json
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Create SQLite engine
engine = create_engine('sqlite:///patient_database.db', echo=True)
Session = sessionmaker(bind=engine)
session = Session()
# Base class for declarative class definitions
Base = declarative_base()

# Define Patient ORM model
class Patient(Base):
    __tablename__ = 'patients'

    id = Column(Integer, primary_key=True)
    patient_name = Column(String)
    date_of_birth = Column(String)
    gender = Column(String)
    blood_type = Column(String)
    height = Column(String)
    weight = Column(String)
    systolic_bp = Column(String)
    diastolic_bp = Column(String)
    heart_rate = Column(String)
    temperature = Column(String)
    reports = relationship("Report", back_populates="patient")

# Define Report ORM model
class Report(Base):
    __tablename__ = 'reports'

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('patients.id'))
    date = Column(String)
    description = Column(String)
    findings = Column(String)
    recommendations = Column(String)
    patient = relationship("Patient", back_populates="reports")

# Create tables in the database
Base.metadata.create_all(engine)

def get_patient_yaml(patient_name_to_search):
    patient_data = session.query(Patient).filter_by(patient_name=patient_name_to_search).first()
    if patient_data:
        patient_dict = {
            "patient_name": patient_data.patient_name,
            "date_of_birth": patient_data.date_of_birth,
            "gender": patient_data.gender,
            "blood_type": patient_data.blood_type,
            "height": patient_data.height,
            "weight": patient_data.weight,
            "blood_pressure": {
                "systolic": patient_data.systolic_bp,
                "diastolic": patient_data.diastolic_bp
            },
            "heart_rate": patient_data.heart_rate,
            "temperature": patient_data.temperature,
            "reports": []
        }

        # Add reports associated with the patient to the dictionary
        for report in patient_data.reports:
            report_dict = {
                "date": report.date,
                "description": report.description,
                "findings": report.findings,
                "recommendations": report.recommendations.split(", ")
            }
            patient_dict["reports"].append(report_dict)
    else:
        patient_dict = {}
    patient_info_yaml = yaml.dump(patient_dict)
    return patient_info_yaml

