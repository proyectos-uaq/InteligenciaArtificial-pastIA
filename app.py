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
    try:
        # Si es formato ISO con Z (UTC)
        if 'T' in time_str and 'Z' in time_str:
            # Remover la Z y convertir a datetime
            time_str = time_str.replace('Z', '+00:00')
            dt = datetime.fromisoformat(time_str)
            # Convertir a naive datetime (sin zona horaria)
            return dt.replace(tzinfo=None)
            
        # Si es solo hora (HH:MM:SS)
        if len(time_str.split(':')) == 3 and 'T' not in time_str:
            hour, minute, second = map(int, time_str.split(':'))
            return datetime.combine(datetime.now().date(), time(hour, minute, second))
            
        # Si es fecha completa en formato ISO
        dt = datetime.fromisoformat(time_str)
        # Convertir a naive datetime si tiene zona horaria
        return dt.replace(tzinfo=None) if dt.tzinfo else dt
    except Exception as e:
        print(f"Error al parsear tiempo: {time_str}, error: {str(e)}")
        return datetime.now()

def calculate_future_schedules(medication, schedule, days=3):
    """Calcula los horarios futuros basados en el patrón de toma y compensa el desfase"""
    future_schedules = []
    
    # Obtener fecha y hora base
    start_date = medication.get('start_date')
    if not start_date:
        return future_schedules
        
    base_time = parse_time(schedule['scheduled_time'])
    
    # Obtener el intervalo del medicamento
    interval_str = medication.get('interval')
    if not interval_str:
        return future_schedules
        
    # Convertir intervalo a horas
    try:
        interval_time = parse_time(interval_str)
        interval_hours = interval_time.hour + (interval_time.minute / 60) + (interval_time.second / 3600)
        # Calcular número de tomas por día (asegurarse de que sea un entero)
        takes_per_day = int(24 / interval_hours)
    except:
        return future_schedules
    
    # Calcular el desfase promedio basado en los logs
    time_diffs = []
    if schedule['intake_logs']:
        # Ordenar logs por tiempo
        sorted_logs = sorted(schedule['intake_logs'], key=lambda x: parse_time(x['time']))
        
        # Calcular diferencias entre cada intake_log y su scheduled_time correspondiente
        for log in sorted_logs:
            log_time = parse_time(log['time'])
            # Calcular cuántas horas han pasado desde el horario base
            hours_since_base = (log_time - base_time).total_seconds() / 3600
            # Calcular el ciclo al que pertenece esta toma
            cycle = int(hours_since_base / interval_hours)
            # Calcular el horario programado para este ciclo
            expected_time = base_time + timedelta(hours=cycle * interval_hours)
            # Calcular la diferencia real
            time_diff = (log_time - expected_time).total_seconds() / 3600
            time_diffs.append(time_diff)
    
    # Calcular el ajuste promedio
    if time_diffs:
        avg_adjustment = sum(time_diffs) / len(time_diffs)
    else:
        avg_adjustment = 0
    
    # Obtener la última fecha de toma de los logs
    last_intake_date = None
    if schedule['intake_logs']:
        last_intake = max(schedule['intake_logs'], key=lambda x: parse_time(x['time']))
        last_intake_date = parse_time(last_intake['time'])
    
    # Si no hay logs, usar la fecha de inicio
    if not last_intake_date:
        last_intake_date = datetime.combine(
            datetime.strptime(start_date, '%Y-%m-%d').date(),
            base_time.time()
        )
    
    # Obtener todos los schedule_ids disponibles para este medicamento
    available_schedule_ids = [s['schedule_id'] for s in medication['schedules']]
    if not available_schedule_ids:
        return future_schedules
    
    # Calcular horarios futuros usando el intervalo ajustado
    current_datetime = last_intake_date
    
    # Solo generar los horarios necesarios para completar el ciclo diario
    for i in range(takes_per_day):
        # Rotar entre los schedule_ids disponibles
        schedule_id = available_schedule_ids[i % len(available_schedule_ids)]
        
        # Aplicar el ajuste promedio al horario base
        adjusted_time = current_datetime + timedelta(hours=avg_adjustment)
        
        future_schedules.append({
            'schedule_id': schedule_id,
            'scheduled_time': adjusted_time.strftime('%H:%M:%S')
        })
        current_datetime += timedelta(hours=interval_hours)
    
    return future_schedules

@app.route('/process', methods=['POST'])
def process_data():
    """Endpoint para procesar datos, entrenar y ajustar horarios"""
    try:
        data = request.get_json()
        
        # Verificar que los datos necesarios estén presentes
        if not data or 'medications' not in data:
            return jsonify({'error': 'Faltan datos requeridos'}), 400
        
        # Verificar que cada medicamento tenga los campos necesarios
        for medication in data['medications']:
            if not all(key in medication for key in ['medication_id', 'start_date', 'interval', 'schedules']):
                return jsonify({'error': 'Faltan campos requeridos en el medicamento'}), 400
            
            for schedule in medication['schedules']:
                if not all(key in schedule for key in ['schedule_id', 'scheduled_time']):
                    return jsonify({'error': 'Faltan campos requeridos en el schedule'}), 400
        
        # Procesar los datos
        success = processor.load_from_db(data)
        if not success:
            return jsonify({'error': 'Error al procesar los datos'}), 500
        
        # Preprocesar datos
        training_data = processor.preprocess_data()
        
        # Preparar secuencias
        sequences = processor.get_training_sequences(training_data)
        
        # Guardar datos procesados
        output_file = 'processed_data.json'
        processor.save_processed_data(sequences, output_file)
        
        # Iniciar entrenamiento automáticamente
        global agent
        
        # Configuración por defecto del agente
        agent_config = {
            'algorithm': 'PPO',
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'verbose': 1
        }
        
        # Inicializar y entrenar el agente
        agent = SmartPillboxAgent(agent_config)
        model_path = agent.train(total_timesteps=100000)
        
        # Evaluar modelo
        mean_reward, std_reward = agent.evaluate()
        
        # Calcular horarios futuros
        response_data = []
        
        # Extraer horarios de los datos procesados
        for medication in data['medications']:
            # Usar el primer schedule para calcular los horarios
            if medication['schedules']:
                schedule = medication['schedules'][0]
                future_times = calculate_future_schedules(medication, schedule)
                
                if future_times:
                    response_data.append({
                        'medication_id': medication['medication_id'],
                        'future_schedules': future_times
                    })
        
        # Crear directorio data si no existe
        os.makedirs('data', exist_ok=True)
        
        # Guardar horarios futuros
        future_file = 'data/future_schedules.json'
        with open(future_file, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False)
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)