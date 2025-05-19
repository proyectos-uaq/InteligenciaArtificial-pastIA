import argparse
import json
import os
from datetime import datetime, timedelta

from environment import SmartPillboxEnv
from agent import SmartPillboxAgent
from data_processor import DataProcessor

def train_model(config_file=None):
    """Entrena un modelo de RL para el pastillero inteligente"""
    config = {}
    if config_file:
        with open(config_file, 'r') as f:
            config = json.load(f)
    
    print("Iniciando entrenamiento del modelo de RL...")
    agent = SmartPillboxAgent(config.get('agent_config'))
    model_path = agent.train(total_timesteps=config.get('total_timesteps', 100000))
    
    # Evaluar el modelo entrenado
    mean_reward, std_reward = agent.evaluate(n_eval_episodes=10)
    
    print(f"Entrenamiento completado. Modelo guardado en: {model_path}")
    print(f"Evaluación del modelo: Recompensa media = {mean_reward:.2f} ± {std_reward:.2f}")
    
    return model_path

def load_and_process_data(schedule_file, intake_file, medication_file=None):
    """Carga y procesa datos para el modelo de RL"""
    processor = DataProcessor()
    
    # Cargar datos
    print(f"Cargando datos desde archivos JSON...")
    processor.load_from_json(schedule_file, intake_file, medication_file)
    
    # Preprocesar datos
    print("Preprocesando datos...")
    training_data = processor.preprocess_data()
    
    # Preparar secuencias de entrenamiento
    print("Preparando secuencias de entrenamiento...")
    sequences = processor.get_training_sequences(training_data)
    
    # Guardar datos procesados
    output_file = 'processed_data.json'
    processor.save_processed_data(sequences, output_file)
    
    print(f"Datos procesados guardados en: {output_file}")
    print(f"Total de secuencias de entrenamiento: {len(sequences)}")
    
    return sequences

def adjust_schedules(model_path, schedule_file, output_file, config_file=None):
    """Ajusta los horarios utilizando el modelo entrenado"""
    config = {}
    if config_file:
        with open(config_file, 'r') as f:
            config = json.load(f)
    
    # Cargar horarios existentes
    with open(schedule_file, 'r') as f:
        schedules = json.load(f)
    
    # Cargar datos históricos de tomas (si están disponibles)
    # En una implementación real, esto vendría de la base de datos
    history_file = config.get('history_file', 'intake_history.json')
    history = []
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    
    # Inicializar agente y cargar modelo entrenado
    agent = SmartPillboxAgent(config.get('agent_config'))
    agent.load(model_path)
    
    # Procesar y ajustar cada horario
    adjusted_schedules = []
    
    print("Ajustando horarios con el modelo de RL...")
    for schedule in schedules:
        # Extraer tiempo programado
        scheduled_time = datetime.fromisoformat(schedule['scheduled_time'])
        
        # Obtener historial relevante para este medicamento
        med_history = [h['compliance'] for h in history 
                      if h.get('medication_id_FK') == schedule.get('medication_id_FK')]
        
        # Ajustar horario
        adjusted_time, adjustment = agent.adjust_schedule(scheduled_time, med_history)
        
        # Crear registro ajustado
        adjusted_schedule = schedule.copy()
        adjusted_schedule['original_time'] = schedule['scheduled_time']
        adjusted_schedule['scheduled_time'] = adjusted_time.isoformat()
        adjusted_schedule['adjustment_minutes'] = adjustment
        
        adjusted_schedules.append(adjusted_schedule)
        
        print(f"Horario ajustado: {scheduled_time} → {adjusted_time} (ajuste: {adjustment:.1f} min)")
    
    # Guardar horarios ajustados
    with open(output_file, 'w') as f:
        json.dump(adjusted_schedules, f, indent=2)
    
    print(f"Horarios ajustados guardados en: {output_file}")
    return adjusted_schedules

def generate_sample_data():
    """Genera datos de muestra para pruebas"""
    # Crear directorios para datos si no existen
    os.makedirs('data', exist_ok=True)
    
    # Generar datos de medicamentos
    medications = [
        {
            "medication_id": f"med_{i}",
            "name": f"Medication {i}",
            "dosage": f"{(i+1)*50}mg",
            "instructions": f"Take with {'food' if i % 2 == 0 else 'water'}",
            "form": f"{'tablet' if i % 3 == 0 else 'pill'}"
        } for i in range(5)
    ]
    
    with open('data/medications.json', 'w') as f:
        json.dump(medications, f, indent=2)
    
    # Generar datos de horarios
    now = datetime.now()
    schedules = []
    
    for med_idx, med in enumerate(medications):
        # Generar horarios para los próximos 14 días
        for day in range(14):
            # Algunos medicamentos se toman varias veces al día
            times_per_day = (med_idx % 3) + 1
            
            for time_idx in range(times_per_day):
                hour = 8 + time_idx * (12 // times_per_day)
                schedule_time = now + timedelta(days=day, hours=hour)
                
                schedules.append({
                    "schedule_id": f"sched_{len(schedules)}",
                    "medication_id_FK": med["medication_id"],
                    "scheduled_time": schedule_time.isoformat()
                })
    
    with open('data/schedules.json', 'w') as f:
        json.dump(schedules, f, indent=2)
    
    # Generar datos de tomas (simular comportamiento del usuario)
    intakes = []
    
    for idx, schedule in enumerate(schedules):
        if idx < len(schedules) * 0.7:  # Solo generar historial para el 70% de los horarios
            scheduled_time = datetime.fromisoformat(schedule["scheduled_time"])
            
            # Simular variabilidad en las tomas
            random_offset = (idx % 7) - 3  # -3 a 3 horas
            variance_minutes = random_offset * 20  # -60 a 60 minutos
            
            # 10% de probabilidad de no tomar la medicación
            is_taken = (idx % 10) != 0
            
            if is_taken:
                button_press_time = scheduled_time + timedelta(minutes=variance_minutes)
                
                intakes.append({
                    "intake_log_id": f"intake_{len(intakes)}",
                    "schedule_id_FK": schedule["schedule_id"],
                    "button_press_time": button_press_time.isoformat(),
                    "is_taken": is_taken
                })
    
    with open('data/intakes.json', 'w') as f:
        json.dump(intakes, f, indent=2)
    
    print("Datos de muestra generados en el directorio 'data':")
    print(f"- Medicamentos: {len(medications)}")
    print(f"- Horarios programados: {len(schedules)}")
    print(f"- Registros de tomas: {len(intakes)}")

def main():
    parser = argparse.ArgumentParser(description="Sistema de pastillero inteligente con RL")
    
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")
    
    # Comando para generar datos de muestra
    gen_parser = subparsers.add_parser("generate", help="Generar datos de muestra")
    
    # Comando para procesar datos
    process_parser = subparsers.add_parser("process", help="Procesar datos para entrenamiento")
    process_parser.add_argument("--schedule", type=str, default="data/schedules.json", 
                              help="Archivo JSON con datos de horarios")
    process_parser.add_argument("--intake", type=str, default="data/intakes.json", 
                              help="Archivo JSON con datos de tomas")
    process_parser.add_argument("--medication", type=str, default="data/medications.json", 
                              help="Archivo JSON con datos de medicamentos")
    
    # Comando para entrenar modelo
    train_parser = subparsers.add_parser("train", help="Entrenar modelo de RL")
    train_parser.add_argument("--config", type=str, default=None, 
                            help="Archivo de configuración para el entrenamiento")
    
    # Comando para ajustar horarios
    adjust_parser = subparsers.add_parser("adjust", help="Ajustar horarios con modelo entrenado")
    adjust_parser.add_argument("--model", type=str, required=True, 
                             help="Ruta al modelo entrenado")
    adjust_parser.add_argument("--schedule", type=str, default="data/schedules.json", 
                             help="Archivo JSON con horarios a ajustar")
    adjust_parser.add_argument("--output", type=str, default="data/adjusted_schedules.json", 
                             help="Archivo de salida para horarios ajustados")
    adjust_parser.add_argument("--config", type=str, default=None, 
                             help="Archivo de configuración")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        generate_sample_data()
    
    elif args.command == "process":
        load_and_process_data(args.schedule, args.intake, args.medication)
    
    elif args.command == "train":
        train_model(args.config)
    
    elif args.command == "adjust":
        adjust_schedules(args.model, args.schedule, args.output, args.config)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 