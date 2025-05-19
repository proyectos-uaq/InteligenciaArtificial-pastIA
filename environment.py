import gym
import numpy as np
from gym import spaces
from datetime import datetime, timedelta
import json

class SmartPillboxEnv(gym.Env):
    """
    Entorno de RL para un pastillero inteligente que ajusta horarios
    basado en los patrones de toma de medicamentos del usuario.
    """
    
    def __init__(self, config=None):
        super(SmartPillboxEnv, self).__init__()
        
        # Configuración por defecto si no se proporciona
        self.config = {
            'max_time_shift': 120,  # Máximo tiempo en minutos para ajustar
            'reward_on_time': 10,   # Recompensa por toma a tiempo
            'penalty_missed': -20,  # Penalización por dosis perdida
            'max_episodes': 100     # Número máximo de episodios
        }
        
        if config:
            self.config.update(config)
        
        # Espacio de acciones: ajuste de tiempo en minutos (-max_shift a +max_shift)
        self.action_space = spaces.Box(
            low=-self.config['max_time_shift'],
            high=self.config['max_time_shift'],
            shape=(1,),
            dtype=np.float32
        )
        
        # Espacio de observación: [diferencia_tiempo_actual, histórico_tomas, hora_del_día]
        self.observation_space = spaces.Box(
            low=np.array([-self.config['max_time_shift'], 0, 0]),
            high=np.array([self.config['max_time_shift'], 1, 24]),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """Reinicia el entorno al inicio de un nuevo episodio"""
        self.current_episode = 0
        self.medication_schedule = []
        self.intake_history = []
        
        # Estado inicial
        self.current_state = np.array([0, 0.5, 12], dtype=np.float32)  # No hay diferencia, historial neutro, mediodía
        return self.current_state
    
    def step(self, action):
        """
        Ejecuta un paso en el entorno aplicando la acción seleccionada
        
        Args:
            action: Ajuste de tiempo en minutos
            
        Returns:
            observation: El nuevo estado
            reward: La recompensa obtenida
            done: Si el episodio ha terminado
            info: Información adicional
        """
        # Aplicar ajuste de tiempo
        time_adjustment = float(action[0])
        
        # Obtener el próximo horario programado
        next_scheduled_time = self.get_next_scheduled_time()
        
        # Ajustar el horario
        adjusted_time = next_scheduled_time + timedelta(minutes=time_adjustment)
        
        # Simular la toma del medicamento (en un sistema real, esto sería el registro real)
        actual_intake_time = self.simulate_intake(adjusted_time)
        
        # Calcular diferencia entre tiempo ajustado y tiempo real de toma
        time_diff = (actual_intake_time - adjusted_time).total_seconds() / 60
        
        # Actualizar historial
        compliance = 1.0 if abs(time_diff) < 30 else max(0, 1 - (abs(time_diff) / 120))
        self.intake_history.append(compliance)
        
        # Calcular recompensa
        reward = self.calculate_reward(time_diff, compliance)
        
        # Actualizar estado
        hour_of_day = actual_intake_time.hour + actual_intake_time.minute / 60.0
        history_factor = np.mean(self.intake_history[-5:]) if len(self.intake_history) >= 5 else 0.5
        self.current_state = np.array([time_diff, history_factor, hour_of_day], dtype=np.float32)
        
        # Verificar si el episodio ha terminado
        self.current_episode += 1
        done = self.current_episode >= self.config['max_episodes']
        
        info = {
            'scheduled_time': next_scheduled_time,
            'adjusted_time': adjusted_time,
            'actual_intake_time': actual_intake_time,
            'compliance': compliance
        }
        
        return self.current_state, reward, done, info
    
    def get_next_scheduled_time(self):
        """
        Obtiene el próximo horario programado
        En un sistema real, esto obtendría datos de la base de datos
        """
        # Simulación de horarios
        if not self.medication_schedule:
            now = datetime.now()
            self.medication_schedule = [
                now + timedelta(hours=i*6) for i in range(1, 5)
            ]
        
        return self.medication_schedule.pop(0)
    
    def simulate_intake(self, scheduled_time):
        """
        Simula cuándo el usuario realmente toma la medicación
        En un sistema real, esto vendría del registro real del pastillero
        """
        # Simular variabilidad en la toma (ruido aleatorio)
        random_variance = np.random.normal(0, 30)  # Desviación estándar de 30 minutos
        return scheduled_time + timedelta(minutes=random_variance)
    
    def calculate_reward(self, time_diff, compliance):
        """Calcula la recompensa basada en la diferencia de tiempo y el cumplimiento"""
        if abs(time_diff) < 15:
            # Toma casi perfecta
            return self.config['reward_on_time']
        elif abs(time_diff) < 60:
            # Toma aceptable
            return self.config['reward_on_time'] * (1 - abs(time_diff) / 60)
        else:
            # Toma muy retrasada o adelantada
            return self.config['penalty_missed'] * (abs(time_diff) / 120)
    
    def load_data_from_db(self, schedule_entries, intake_logs):
        """
        Carga datos reales de la base de datos
        
        Args:
            schedule_entries: Lista de entradas de horarios programados
            intake_logs: Lista de registros de tomas reales
        """
        # Implementar lógica para cargar datos reales
        pass 