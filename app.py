from flask import Flask, request, jsonify
from data_processor import DataProcessor
from agent import SmartPillboxAgent
import json
import os
from datetime import datetime, time, timedelta

app = Flask(__name__)

# Inicializar el procesador de datos y el agente
processor = DataProcessor()
agent = None  # Se inicializará cuando se entrene el modelo

def parse_time(time_str):
    """Convierte una cadena de tiempo a datetime"""
    if not time_str:
        return datetime.now()
        
    try:
        # Si es formato ISO con Z (UTC)
        if 'T' in time_str and 'Z' in time_str:
            time_str = time_str.replace('Z', '+00:00')
            
        # Si es solo hora (HH:MM:SS)
        if len(time_str.split(':')) == 3 and 'T' not in time_str:
            hour, minute, second = map(int, time_str.split(':'))
            return datetime.combine(datetime.now().date(), time(hour, minute, second))
            
        # Si es fecha completa en formato ISO
        dt = datetime.fromisoformat(time_str)
        return dt.replace(tzinfo=None) if dt.tzinfo else dt
        
    except Exception as e:
        print(f"Error al parsear tiempo: {time_str}, error: {str(e)}")
        return datetime.now()

def validate_medication_data(data):
    """Valida que los datos del medicamento tengan todos los campos requeridos"""
    if not data or 'medications' not in data:
        return False, 'Faltan datos requeridos'
    
    required_med_fields = ['medication_id', 'start_date', 'interval', 'schedules']
    required_schedule_fields = ['schedule_id', 'scheduled_time']
    
    for medication in data['medications']:
        if not all(key in medication for key in required_med_fields):
            return False, 'Faltan campos requeridos en el medicamento'
        
        for schedule in medication['schedules']:
            if not all(key in schedule for key in required_schedule_fields):
                return False, 'Faltan campos requeridos en el schedule'
    
    return True, None

def calculate_future_schedules(medication, schedule, days=3):
    """Calcula los horarios futuros basados en el patrón de toma y compensa el desfase"""
    try:
        if not medication.get('start_date') or not medication.get('interval'):
            return []
            
        base_time = parse_time(schedule['scheduled_time'])
        interval_time = parse_time(medication['interval'])
        interval_hours = interval_time.hour + (interval_time.minute / 60) + (interval_time.second / 3600)
        takes_per_day = int(24 / interval_hours)
        
        # Calcular el desfase promedio basado en los logs
        time_diffs = []
        if schedule.get('intake_logs'):
            sorted_logs = sorted(schedule['intake_logs'], key=lambda x: parse_time(x['time']))
            
            for log in sorted_logs:
                log_time = parse_time(log['time'])
                hours_since_base = (log_time - base_time).total_seconds() / 3600
                cycle = int(hours_since_base / interval_hours)
                expected_time = base_time + timedelta(hours=cycle * interval_hours)
                time_diff = (log_time - expected_time).total_seconds() / 3600
                time_diffs.append(time_diff)
        
        avg_adjustment = sum(time_diffs) / len(time_diffs) if time_diffs else 0
        
        # Obtener la última fecha de toma
        last_intake_date = None
        if schedule.get('intake_logs'):
            last_intake = max(schedule['intake_logs'], key=lambda x: parse_time(x['time']))
            last_intake_date = parse_time(last_intake['time'])
        else:
            last_intake_date = datetime.combine(
                datetime.strptime(medication['start_date'], '%Y-%m-%d').date(),
                base_time.time()
            )
        
        # Obtener schedule_ids disponibles
        available_schedule_ids = [s['schedule_id'] for s in medication['schedules']]
        if not available_schedule_ids:
            return []
        
        # Calcular horarios futuros
        future_schedules = []
        current_datetime = last_intake_date
        
        for i in range(takes_per_day):
            schedule_id = available_schedule_ids[i % len(available_schedule_ids)]
            adjusted_time = current_datetime + timedelta(hours=avg_adjustment)
            
            future_schedules.append({
                'schedule_id': schedule_id,
                'scheduled_time': adjusted_time.strftime('%H:%M:%S')
            })
            current_datetime += timedelta(hours=interval_hours)
        
        return future_schedules
        
    except Exception as e:
        print(f"Error al calcular horarios futuros: {str(e)}")
        return []

def save_to_json(data, filename):
    """Guarda datos en un archivo JSON"""
    try:
        dirpath = os.path.dirname(filename)
        if dirpath:  # Solo crea el directorio si no es vacío
            os.makedirs(dirpath, exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error al guardar archivo {filename}: {str(e)}")
        return False

@app.route('/process', methods=['POST'])
def process_data():
    """Endpoint para procesar datos, entrenar y ajustar horarios"""
    try:
        data = request.get_json()
        
        # Validar datos
        is_valid, error_msg = validate_medication_data(data)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Procesar datos
        if not processor.load_from_db(data):
            return jsonify({'error': 'Error al procesar los datos'}), 500
        
        # Preprocesar y preparar datos
        training_data = processor.preprocess_data()
        sequences = processor.get_training_sequences(training_data)
        
        # Guardar datos procesados
        if not save_to_json(sequences, 'processed_data.json'):
            return jsonify({'error': 'Error al guardar datos procesados'}), 500
        
        # Iniciar entrenamiento
        global agent
        agent = SmartPillboxAgent()
        model_path = agent.train(total_timesteps=100000)
        
        # Evaluar modelo
        mean_reward, std_reward = agent.evaluate()
        
        # Calcular horarios futuros
        response_data = []
        for medication in data['medications']:
            if medication['schedules']:
                schedule = medication['schedules'][0]
                future_times = calculate_future_schedules(medication, schedule)
                
                if future_times:
                    response_data.append({
                        'medication_id': medication['medication_id'],
                        'future_schedules': future_times
                    })
        
        # Guardar horarios futuros
        if not save_to_json(response_data, 'data/future_schedules.json'):
            return jsonify({'error': 'Error al guardar horarios futuros'}), 500
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)