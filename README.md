# Pastillero Inteligente con Reinforcement Learning

Sistema inteligente que ajusta automáticamente los horarios de medicación basado en patrones de toma utilizando técnicas de Reinforcement Learning.

## Descripción

Este sistema implementa un agente de RL que aprende a ajustar los horarios de medicación para adaptarse a los patrones de comportamiento del usuario. Si el usuario tiende a tomar medicamentos tarde o temprano, el sistema ajustará los recordatorios para maximizar la adherencia al tratamiento.

## Características

- Ajuste automático de horarios basado en patrones de toma anteriores
- Análisis de datos históricos para predecir comportamientos
- Recompensas por tomar medicamentos a tiempo
- Adaptación dinámica a cambios en los patrones de comportamiento

## Estructura de la Base de Datos

El sistema utiliza las siguientes tablas:

- **Users**: Información de usuarios
- **Prescriptions**: Recetas médicas vinculadas a usuarios
- **Medications**: Medicamentos vinculados a recetas
- **Schedule_entries**: Entradas de horario para medicamentos
- **Intake_logs**: Registros de toma de medicamentos

## Requisitos

```
pip install -r requirements.txt
```

## Uso

### Generar datos de muestra

```bash
python main.py generate
```

### Procesar datos para entrenamiento

```bash
python main.py process --schedule data/schedules.json --intake data/intakes.json --medication data/medications.json
```

### Entrenar modelo

```bash
python main.py train --config config.json
```

### Ajustar horarios con modelo entrenado

```bash
python main.py adjust --model models/pillbox_PPO --schedule data/schedules.json --output data/adjusted_schedules.json
```

## Configuración

Puedes personalizar el comportamiento del sistema con un archivo de configuración JSON:

```json
{
  "agent_config": {
    "algorithm": "PPO",
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "n_steps": 2048,
    "ent_coef": 0.01
  },
  "total_timesteps": 100000,
  "history_file": "data/intake_history.json"
}
```

## Cómo funciona

1. El sistema recopila datos de las tomas de medicamentos del usuario
2. Preprocesa estos datos para extraer patrones de comportamiento
3. Entrena un modelo de RL que aprende a predecir cuándo es más probable que el usuario tome su medicación
4. Ajusta los horarios programados para maximizar la adherencia al tratamiento

## Algoritmo de RL utilizado

El sistema usa por defecto el algoritmo **PPO (Proximal Policy Optimization)** de Stable Baselines3, que ofrece un buen equilibrio entre rendimiento y estabilidad. También soporta SAC y A2C como alternativas.

## Estructura del Proyecto

- `environment.py`: Entorno de simulación para el RL
- `agent.py`: Implementación del agente de RL
- `data_processor.py`: Procesamiento de datos históricos
- `main.py`: Punto de entrada principal
- `data/`: Directorio para datos de entrada/salida
- `models/`: Directorio para modelos entrenados
- `logs/`: Directorio para registros de entrenamiento 