import tensorflow as tf
import numpy as np
from snake import Game
from collections import deque
import random

# Параметры Q-обучения
gamma = 0.9  # коэффициент дисконтирования
epsilon = 0.45  # параметр epsilon для epsilon-greedy стратегии
learning_rate = 0.001
# Размер входных данных [danger_left, danger_straight, danger_right,
# direction_left, direction_up, direction_right, direction_down,
# food_left, food_up, food_right, food_down]
input_size = 11
# Количество возможных действий
# [1, 0, 0] - "left"
# [0, 1, 0] - "straight"
# [0, 0, 1] - "right"
output_size = 3
experience_replay_buffer = deque(maxlen=10000)
total_reward = 0
max_score = 0

#     tf.keras.layers.Dense(256, input_dim=input_size, activation='relu'),
#tf.keras.layers.Dense(64, activation='relu', input_shape=(input_size,)),
#tf.keras.layers.Dense(32, activation='relu'),
# Задаем структуру нейронной сети

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(input_size,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='linear')
])
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

#model = tf.keras.models.load_model("trained_model_5.keras")

# Функция для преобразования действия в целевое значение
def action_to_target(action):
    if action == 'left':
        return np.array([1, 0, 0])
    elif action == 'straight':
        return np.array([0, 1, 0])
    elif action == 'right':
        return np.array([0, 0, 1])

def AI_direction(state):
    # Выбор действия с использованием epsilon-жадной стратегии
    if random.random() < epsilon:
        action = np.random.choice(['left', 'straight', 'right'])
        target = np.array([action_to_target(action)])
    else:
        target = model.predict(np.array([state]))
        action_index = np.argmax(target)
        directions = ["left", "straight", "right"]
        action = directions[action_index]
    #print("Predicted Direction: ", action, " | target: ", target)
    return action, target

# Обучение нейронной сети с помощью Q-learning
def AI_train(last_state, action, target, new_state, reward):
    last_state = np.array([last_state])
    target = np.array(target)
    reward = reward
    new_state = np.array([new_state])
    # Вычисление целевого Q-значения
    action_index = np.argmax(action_to_target(action))
    # target = model.predict(last_state)
    # Обновление целевого значения с учетом Q-обучения
    next_q_values = model.predict(new_state)
    # target[0][action_index] = (1 - learning_rate) * target[0][action_index] + learning_rate * (reward + gamma * np.max(next_q_values))
    target[0][action_index] = reward + gamma * np.max(next_q_values)
    # Обучение нейронной сети на текущем состоянии и целевом значении
    # model.fit(last_state, target, epochs=1, verbose=0)
    model.train_on_batch(last_state, target)
    #print("state: ", last_state, " | target: ", target)

# Функция для обучения на случайном наборе прошлых опытов
def train_on_experience_batch(batch_size=64):
    if len(experience_replay_buffer) < batch_size:
        return
    batch = random.sample(experience_replay_buffer, batch_size)
    for experience in batch:
        last_state, action, reward, new_state = experience
        AI_train(last_state, action, target, new_state, reward)

for probe in range(150):
    game = Game()
    # Обучение с использованием Q-learning
    game.start_game("The Snake Game 1.0 | probe:" + str(probe) + " | epsilon: " + str(round(epsilon, 3)) + " | max_score: " + str(max_score))
    while not game.game_over:
        state = game.get_state()
        action, target = AI_direction(state)
        reward = game.step(action)
        new_state = game.get_state()
        game.repaint()
        # Добавление опыта в буфер
        # experience_replay_buffer.append((state, action, reward, new_state))
        AI_train(state, action, target, new_state, reward)
        game.wait()
    # Обучение на случайной выборке из буфера опыта
    #train_on_experience_batch()
    score = game.stop_game()
    if score > max_score:
        max_score = score

    # Уменьшение epsilon по мере обучения
    if epsilon >= 0.01:
        epsilon -= 0.006

print("last score: " + str(score) + " | max score: " + str(max_score) + " | epsilon: " + str(round(epsilon, 3)))

# Сохранение модели
model.save("trained_model_5.keras")
