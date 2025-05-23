# Pastillero Inteligente con Reinforcement Learning

Sistema inteligente que ajusta automáticamente los horarios de medicación basado en patrones de toma utilizando técnicas de Reinforcement Learning y una API REST.

## Descripción

Este sistema implementa un agente de RL que aprende a ajustar los horarios de medicación para adaptarse a los patrones de comportamiento del usuario. Si el usuario tiende a tomar medicamentos tarde o temprano, el sistema ajustará los recordatorios para maximizar la adherencia al tratamiento.

## Características

- Ajuste automático de horarios basado en patrones de toma anteriores
- Análisis de datos históricos para predecir comportamientos
- Recompensas por tomar medicamentos a tiempo
- Adaptación dinámica a cambios en los patrones de comportamiento
- API REST para integración con otros sistemas

## Estructura de la Base de Datos (referencial)

El sistema espera datos estructurados con la siguiente lógica:

- **Users**: Información de usuarios
- **Prescriptions**: Recetas médicas vinculadas a usuarios
- **Medications**: Medicamentos vinculados a recetas
- **Schedule_entries**: Entradas de horario para medicamentos
- **Intake_logs**: Registros de toma de medicamentos

## Requisitos

```bash
pip install -r requirements.txt
```

## Uso de la API

1. **Inicia el servidor Flask:**

```bash
python app.py
```

2. **Envía una solicitud POST a `/process` con el siguiente formato JSON:**

```json
{
  "medications": [
    {
      "medication_id": "med1",
      "start_date": "2024-06-01",
      "interval": "08:00:00",
      "schedules": [
        {"schedule_id": "sch1", "scheduled_time": "08:00:00", "intake_logs": [
          {"time": "2024-06-01T08:10:00"},
          {"time": "2024-06-02T08:05:00"}
        ]}
      ]
    }
  ]
}
```

3. **La respuesta será un JSON con los próximos horarios ajustados:**

```json
[
  {
    "medication_id": "med1",
    "future_schedules": [
      {"schedule_id": "sch1", "scheduled_time": "08:07:00"},
      {"schedule_id": "sch1", "scheduled_time": "16:07:00"},
      ...
    ]
  }
]
```

## Flujo del sistema

1. El usuario o sistema externo envía los datos de medicamentos y tomas vía API.
2. El sistema procesa y preprocesa los datos reales.
3. Se entrena automáticamente el modelo PPO.
4. Se calculan y devuelven los horarios futuros ajustados según el comportamiento histórico.

## Estructura del Proyecto

- `app.py`: API Flask principal, procesamiento de datos y cálculo de horarios futuros
- `data_processor.py`: Procesamiento y preprocesamiento de datos históricos
- `agent.py`: Implementación y entrenamiento del agente PPO
- `environment.py`: Entorno de RL para el pastillero
- `main.py`: (Obsoleto para uso directo, solo referencia)
- `data/`: Directorio para datos de entrada/salida
- `models/`: Directorio para modelos entrenados
- `logs/`: Directorio para registros de entrenamiento

## Algoritmo de RL utilizado

El sistema usa el algoritmo **PPO (Proximal Policy Optimization)** de Stable Baselines3.

## Notas adicionales

- El sistema está preparado para recibir datos reales y ajustarse automáticamente.
- Los archivos `processed_data.json` y `future_schedules.json` se generan automáticamente.
- Para visualizar métricas de entrenamiento, puedes usar TensorBoard:

```bash
tensorboard --logdir logs/
``` 