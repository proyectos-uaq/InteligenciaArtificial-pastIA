import json
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

class DataProcessor:
    """
    Clase para procesar datos de la base de datos y prepararlos para el modelo de RL
    """
    
    def __init__(self):
        self.schedule_data = None
        self.intake_data = None
        self.medication_data = None
    
    def load_from_json(self, schedule_file, intake_file, medication_file=None):
        """
        Carga datos desde archivos JSON
        
        Args:
            schedule_file: Ruta al archivo JSON con datos de horarios
            intake_file: Ruta al archivo JSON con datos de tomas
            medication_file: Ruta al archivo JSON con datos de medicamentos (opcional)
        """
        with open(schedule_file, 'r') as f:
            self.schedule_data = json.load(f)
        
        with open(intake_file, 'r') as f:
            self.intake_data = json.load(f)
        
        if medication_file:
            with open(medication_file, 'r') as f:
                self.medication_data = json.load(f)
    
    def load_from_db(self, schedule_entries, intake_logs, medications=None):
        """
        Carga datos directamente desde las tablas de la base de datos
        
        Args:
            schedule_entries: Lista de entradas de horarios
            intake_logs: Lista de registros de tomas
            medications: Lista de medicamentos (opcional)
        """
        self.schedule_data = schedule_entries
        self.intake_data = intake_logs
        self.medication_data = medications
    
    def preprocess_data(self):
        """
        Preprocesa los datos para el modelo de RL
        
        Returns:
            training_data: Datos preprocesados para entrenamiento
        """
        if not self.schedule_data or not self.intake_data:
            raise ValueError("Datos no cargados. Primero carga los datos con load_from_json o load_from_db")
        
        # Convertir a DataFrame para facilitar el procesamiento
        schedule_df = pd.DataFrame(self.schedule_data)
        intake_df = pd.DataFrame(self.intake_data)
        
        # Asegurarse de que las columnas esperadas estén presentes
        required_schedule_cols = ["schedule_id", "medication_id_FK", "scheduled_time"]
        required_intake_cols = ["schedule_id_FK", "button_press_time", "is_taken"]
        
        for col in required_schedule_cols:
            if col not in schedule_df.columns:
                raise ValueError(f"Columna requerida '{col}' no encontrada en los datos de horarios")
        
        for col in required_intake_cols:
            if col not in intake_df.columns:
                raise ValueError(f"Columna requerida '{col}' no encontrada en los datos de tomas")
        
        # Convertir tiempos a objetos datetime
        schedule_df["scheduled_time"] = pd.to_datetime(schedule_df["scheduled_time"])
        intake_df["button_press_time"] = pd.to_datetime(intake_df["button_press_time"])
        
        # Unir los datos de horarios y tomas
        merged_data = pd.merge(
            schedule_df,
            intake_df,
            left_on="schedule_id",
            right_on="schedule_id_FK",
            how="left"
        )
        
        # Calcular diferencia de tiempo entre horario programado y toma real (en minutos)
        merged_data["time_diff"] = (merged_data["button_press_time"] - merged_data["scheduled_time"]).dt.total_seconds() / 60
        
        # Calcular hora del día (característica importante para el modelo)
        merged_data["hour_of_day"] = merged_data["scheduled_time"].dt.hour + merged_data["scheduled_time"].dt.minute / 60
        
        # Calcular cumplimiento (1 si se tomó a tiempo, 0 si no)
        merged_data["compliance"] = np.where(
            (merged_data["is_taken"] == True) & (merged_data["time_diff"].abs() < 60),
            1.0,
            np.where(
                merged_data["is_taken"] == True,
                0.5,  # Tomada pero con retraso/adelanto significativo
                0.0   # No tomada
            )
        )
        
        # Ordenar por usuario y tiempo
        if "user_id_FK" in merged_data.columns:
            merged_data = merged_data.sort_values(["user_id_FK", "scheduled_time"])
        else:
            merged_data = merged_data.sort_values(["scheduled_time"])
        
        # Crear características históricas (patrones de cumplimiento previo)
        merged_data["prev_compliance"] = merged_data.groupby(
            "medication_id_FK")["compliance"].shift(1).fillna(0.5)
        
        # Calcular promedio móvil de cumplimiento (últimas 5 tomas)
        merged_data["compliance_ma"] = merged_data.groupby(
            "medication_id_FK")["compliance"].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        
        # Preparar datos de entrenamiento
        training_data = merged_data[[
            "medication_id_FK", "scheduled_time", "button_press_time", 
            "time_diff", "hour_of_day", "compliance", "prev_compliance", 
            "compliance_ma", "is_taken"
        ]].dropna(subset=["time_diff"])
        
        return training_data
    
    def get_training_sequences(self, training_data, sequence_length=10):
        """
        Prepara secuencias de datos para entrenar modelos de RL
        
        Args:
            training_data: DataFrame con datos preprocesados
            sequence_length: Longitud de las secuencias de entrenamiento
            
        Returns:
            sequences: Lista de secuencias para entrenamiento
        """
        sequences = []
        
        # Agrupar por medicamento
        for med_id, group in training_data.groupby("medication_id_FK"):
            # Ordenar por tiempo
            group = group.sort_values("scheduled_time")
            
            # Crear secuencias
            for i in range(len(group) - sequence_length + 1):
                sequence = group.iloc[i:i+sequence_length]
                
                # Convertir a formato adecuado para el entorno de RL
                seq_data = {
                    "medication_id": med_id,
                    "observations": [],
                    "actions": [],
                    "rewards": []
                }
                
                # Calcular observaciones, acciones y recompensas
                for j in range(sequence_length - 1):
                    current = sequence.iloc[j]
                    next_row = sequence.iloc[j+1]
                    
                    # Observación: [diferencia_tiempo, histórico_cumplimiento, hora_del_día]
                    observation = [
                        float(current["time_diff"]),
                        float(current["compliance_ma"]),
                        float(current["hour_of_day"])
                    ]
                    
                    # Acción: ajuste de tiempo (diferencia entre horario programado y el siguiente)
                    time_diff_minutes = (next_row["scheduled_time"] - (current["scheduled_time"] + timedelta(minutes=current["time_diff"]))).total_seconds() / 60
                    action = [float(time_diff_minutes)]
                    
                    # Recompensa: basada en cumplimiento
                    reward = float(next_row["compliance"] * 10)  # Escalar para el entorno
                    
                    seq_data["observations"].append(observation)
                    seq_data["actions"].append(action)
                    seq_data["rewards"].append(reward)
                
                sequences.append(seq_data)
        
        return sequences
    
    def save_processed_data(self, data, output_file):
        """
        Guarda los datos procesados en un archivo JSON
        
        Args:
            data: Datos procesados
            output_file: Ruta al archivo de salida
        """
        # Convertir DataFrame a formato serializable
        if isinstance(data, pd.DataFrame):
            data_dict = data.to_dict(orient="records")
        else:
            data_dict = data
            
        # Convertir fechas a strings
        def convert_dates(obj):
            if isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            return obj
            
        # Guardar como JSON
        with open(output_file, 'w') as f:
            json.dump(data_dict, f, default=convert_dates, indent=2) 