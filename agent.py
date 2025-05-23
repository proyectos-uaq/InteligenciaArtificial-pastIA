from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import json
from environment import SmartPillboxEnv
import os

class SmartPillboxAgent:
    """
    Agente de RL para el pastillero inteligente usando Stable Baselines3
    """
    
    def __init__(self, config=None):
        # Configuración por defecto
        self.config = {
            'learning_rate': 0.0003,  # Tasa de aprendizaje
            'gamma': 0.99,            # Factor de descuento
            'n_steps': 2048,          # Pasos por actualización
            'ent_coef': 0.01,         # Coeficiente de entropía
            'log_dir': './logs/',     # Directorio para logs
            'save_dir': './models/'   # Directorio para guardar modelos
        }
        
        if config:
            self.config.update(config)
        
        # Crear directorios si no existen
        os.makedirs(self.config['log_dir'], exist_ok=True)
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        # Crear entorno
        self.env = DummyVecEnv([lambda: SmartPillboxEnv()])
        
        # Inicializar modelo PPO
        self.model = PPO(
            "MlpPolicy", 
            self.env, 
            learning_rate=self.config['learning_rate'],
            gamma=self.config['gamma'],
            n_steps=self.config['n_steps'],
            ent_coef=self.config['ent_coef'],
            verbose=1,
            tensorboard_log=self.config['log_dir']
        )
    
    def train(self, total_timesteps=100000):
        """
        Entrena el modelo
        
        Args:
            total_timesteps: Número total de pasos de entrenamiento
        """
        self.model.learn(total_timesteps=total_timesteps)
        
        # Guardar el modelo entrenado
        model_path = os.path.join(self.config['save_dir'], "pillbox_ppo")
        self.model.save(model_path)
        print(f"Modelo guardado en {model_path}")
        
        return model_path
    
    def load(self, model_path):
        """
        Carga un modelo previamente entrenado
        
        Args:
            model_path: Ruta al modelo guardado
        """
        self.model = PPO.load(model_path, env=self.env)
        print(f"Modelo cargado desde {model_path}")
    
    def evaluate(self, n_eval_episodes=10):
        """
        Evalúa el modelo entrenado
        
        Args:
            n_eval_episodes: Número de episodios para la evaluación
            
        Returns:
            mean_reward: Recompensa media
            std_reward: Desviación estándar de la recompensa
        """
        mean_reward, std_reward = evaluate_policy(
            self.model, 
            self.env, 
            n_eval_episodes=n_eval_episodes
        )
        
        print(f"Recompensa media: {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward
    
    def adjust_schedule(self, scheduled_time, history):
        """
        Ajusta el horario programado basado en el historial de tomas
        
        Args:
            scheduled_time: Tiempo programado original
            history: Historial de tomas recientes
            
        Returns:
            adjusted_time: Tiempo ajustado recomendado
            time_adjustment: Ajuste aplicado en minutos
        """
        # Preparar observación para el modelo
        # [diferencia_tiempo, histórico_cumplimiento, hora_del_día]
        current_hour = scheduled_time.hour + scheduled_time.minute / 60.0
        
        # Calcular factor de historial (promedio de cumplimiento reciente)
        history_factor = np.mean(history) if history else 0.5
        
        # Observación inicial (sin diferencia de tiempo al inicio)
        observation = np.array([0, history_factor, current_hour], dtype=np.float32)
        
        # Predecir ajuste
        action, _ = self.model.predict(observation, deterministic=True)
        time_adjustment = float(action[0])
        
        # Aplicar ajuste
        from datetime import timedelta
        adjusted_time = scheduled_time + timedelta(minutes=time_adjustment)
        
        return adjusted_time, time_adjustment 