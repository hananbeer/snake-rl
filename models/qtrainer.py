import torch
import torch.nn as nn
import torch.optim as optim

class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        use_scheduler = False
        if use_scheduler:
          self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)
        else:
          self.scheduler = None
        self.criterion = nn.MSELoss()
        self.max_grad_norm = 1.0

    def train_step(self, state, action, reward, next_state, done, use_double_dqn=True):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()

        use_target_model = False
        if use_target_model:
          self.target_model.eval()
          with torch.no_grad():
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    if use_double_dqn:
                        # Double DQN: use main network to select action, target network to evaluate
                        next_state_tensor = next_state[idx].unsqueeze(0)
                        next_state_q_values = self.model(next_state_tensor)
                        best_action = torch.argmax(next_state_q_values, dim=1).item()
                        target_q_value = self.target_model(next_state_tensor)[0][best_action]
                        Q_new = reward[idx] + self.gamma * target_q_value
                    else:
                        # Standard DQN: use target network for both selection and evaluation
                        Q_new = reward[idx] + self.gamma * torch.max(self.target_model(next_state[idx].unsqueeze(0)))
                
                target[idx][torch.argmax(action[idx]).item()] = Q_new
        else:
          for idx in range(len(done)):
              Q_new = reward[idx]
              if not done[idx]:
                  Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

              target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        
        # Gradient clipping
        use_grad_clipping = False
        if use_grad_clipping:
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()

    def update_scheduler(self):
      if self.scheduler:
        self.scheduler.step()



