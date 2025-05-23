import json
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

class DataProcessor:
    """
    Clase para procesar datos de la base de datos y prepararlos para el modelo de RL
    """
    
    def __init__(self):
        self.data = None
        self.processed_data = None
    
    def load_from_json(self, schedule_file, intake_file, medication_file=None):
        """
        Carga datos desde archivos JSON
        
        Args:
            schedule_file: Ruta al archivo JSON con datos de horarios
            intake_file: Ruta al archivo JSON con datos de tomas
            medication_file: Ruta al archivo JSON con datos de medicamentos (opcional)
        """
        with open(schedule_file, 'r') as f:
            self.schedules = json.load(f)
        
        with open(intake_file, 'r') as f:
            self.intakes = json.load(f)
        
        if medication_file:
            with open(medication_file, 'r') as f:
                self.medications = json.load(f)
    
    def load_from_db(self, data):
        """Carga datos desde el JSON recibido"""
        try:
            self.data = data
            return True
        except Exception as e:
            print(f"Error al cargar datos: {str(e)}")
            return False
    
    def preprocess_data(self):
        """Preprocesa los datos para el entrenamiento"""
        if not self.data:
            return None
            
        processed_data = []
        
        for medication in self.data['medications']:
            for schedule in medication['schedules']:
                # Extraer datos básicos
                schedule_data = {
                    'medication_id': medication['medication_id'],
                    'schedule_id': schedule['schedule_id'],
                    'scheduled_time': schedule['scheduled_time'],
                    'start_date': medication['start_date'],
                    'interval': medication['interval']
                }
                
                # Procesar logs de toma
                if 'intake_logs' in schedule:
                    schedule_data['intake_logs'] = schedule['intake_logs']
                else:
                    schedule_data['intake_logs'] = []
                
                processed_data.append(schedule_data)
        
        self.processed_data = processed_data
        return processed_data
    
    def get_training_sequences(self, data):
        """Genera secuencias de entrenamiento"""
        if not data:
            return []
            
        sequences = []
        
        for schedule in data:
            # Crear secuencia base
            sequence = {
                'medication_id': schedule['medication_id'],
                'schedule_id': schedule['schedule_id'],
                'scheduled_time': schedule['scheduled_time'],
                'start_date': schedule['start_date'],
                'interval': schedule['interval'],
                'intake_logs': schedule['intake_logs']
            }
            
            sequences.append(sequence)
        
        return sequences
    
    def save_processed_data(self, data, output_file):
        """Guarda los datos procesados en un archivo JSON"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error al guardar datos: {str(e)}")
            return False 