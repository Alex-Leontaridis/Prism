# This file defines the NASJob and NASResult tables for the NAS backend.
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class NASJob(Base):
    __tablename__ = 'nas_jobs'
    id = Column(Integer, primary_key=True)
    job_id = Column(String, unique=True, index=True)
    config = Column(JSON)
    seed = Column(Integer)
    status = Column(String, default='pending')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    results = relationship('NASResult', back_populates='job')

class NASResult(Base):
    __tablename__ = 'nas_results'
    id = Column(Integer, primary_key=True)
    job_id = Column(String, ForeignKey('nas_jobs.job_id'))
    step = Column(Integer)
    architecture = Column(JSON)
    hparams = Column(JSON)
    metrics = Column(JSON)
    job = relationship('NASJob', back_populates='results') 