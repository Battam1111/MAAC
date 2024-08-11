import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import AttentionAgent
from utils.critics import AttentionCritic

# 均方误差损失函数
MSELoss = torch.nn.MSELoss()

class AttentionSAC(object):
    """
    使用集中注意力的SAC算法的代理类，用于多智能体任务
    """

    def __init__(self, agent_init_params, sa_size, gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10., pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4, **kwargs):
        """
        初始化AttentionSAC类

        参数:
            agent_init_params (list of dict): 初始化每个代理的参数字典列表
                num_in_pol (int): 策略的输入维度
                num_out_pol (int): 策略的输出维度
            sa_size (list of (int, int)): 每个代理的状态和动作空间大小
            gamma (float): 折扣因子
            tau (float): 目标网络更新速率
            pi_lr (float): 策略网络学习率
            q_lr (float): 评论家网络学习率
            reward_scale (float): 奖励缩放（影响最优策略的熵）
            pol_hidden_dim (int): 策略网络的隐藏层维度
            critic_hidden_dim (int): 评论家网络的隐藏层维度
            attend_heads (int): 注意力头的数量
        """
        self.nagents = len(sa_size)  # 智能体的数量

        # 初始化每个智能体的策略网络
        self.agents = [AttentionAgent(lr=pi_lr, hidden_dim=pol_hidden_dim, **params) for params in agent_init_params]
        
        # 初始化评论家网络和目标评论家网络
        self.critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim, attend_heads=attend_heads)
        self.target_critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim, attend_heads=attend_heads)
        
        # 硬更新目标评论家网络
        hard_update(self.target_critic, self.critic)
        
        # 设置评论家网络优化器
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr, weight_decay=1e-3)
        
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.pol_dev = 'cpu'  # 策略网络的设备
        self.critic_dev = 'cpu'  # 评论家网络的设备
        self.trgt_pol_dev = 'cpu'  # 目标策略网络的设备
        self.trgt_critic_dev = 'cpu'  # 目标评论家网络的设备
        self.niter = 0  # 迭代次数

    @property
    def policies(self):
        """
        获取所有智能体的策略网络
        """
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        """
        获取所有智能体的目标策略网络
        """
        return [a.target_policy for a in self.agents]

    def step(self, observations, explore=False):
        """
        所有智能体在环境中前进一步
        参数:
            observations: 每个智能体的观察值列表
            explore (bool): 是否进行探索
        返回:
            actions: 每个智能体的动作列表
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents, observations)]

    def update_critic(self, sample, soft=True, logger=None, **kwargs):
        """
        更新所有智能体的中央评论家网络
        参数:
            sample: 从回放缓冲区采样的数据
            soft (bool): 是否使用软更新
            logger: 日志记录器
        """
        obs, acs, rews, next_obs, dones = sample
        # Q损失
        next_acs = []
        next_log_pis = []
        for pi, ob in zip(self.target_policies, next_obs):
            curr_next_ac, curr_next_log_pi = pi(ob, return_log_pi=True)
            next_acs.append(curr_next_ac)
            next_log_pis.append(curr_next_log_pi)
        trgt_critic_in = list(zip(next_obs, next_acs))
        critic_in = list(zip(obs, acs))
        next_qs = self.target_critic(trgt_critic_in)
        critic_rets = self.critic(critic_in, regularize=True, logger=logger, niter=self.niter)
        q_loss = 0
        for a_i, nq, log_pi, (pq, regs) in zip(range(self.nagents), next_qs, next_log_pis, critic_rets):
            target_q = (rews[a_i].view(-1, 1) + self.gamma * nq * (1 - dones[a_i].view(-1, 1)))
            if soft:
                target_q -= log_pi / self.reward_scale
            q_loss += MSELoss(pq, target_q.detach())
            for reg in regs:
                q_loss += reg  # 正则化注意力
        q_loss.backward()
        self.critic.scale_shared_grads()
        grad_norm = torch.nn.utils.clip_grad_norm(self.critic.parameters(), 10 * self.nagents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('grad_norms/q', grad_norm, self.niter)
        self.niter += 1

    def update_policies(self, sample, soft=True, logger=None, **kwargs):
        """
        更新所有智能体的策略网络
        参数:
            sample: 从回放缓冲区采样的数据
            soft (bool): 是否使用软更新
            logger: 日志记录器
        """
        obs, acs, rews, next_obs, dones = sample
        samp_acs = []
        all_probs = []
        all_log_pis = []
        all_pol_regs = []

        for a_i, pi, ob in zip(range(self.nagents), self.policies, obs):
            curr_ac, probs, log_pi, pol_regs, ent = pi(ob, return_all_probs=True, return_log_pi=True, regularize=True, return_entropy=True)
            if logger is not None:
                logger.add_scalar('agent%i/policy_entropy' % a_i, ent, self.niter)
            samp_acs.append(curr_ac)
            all_probs.append(probs)
            all_log_pis.append(log_pi)
            all_pol_regs.append(pol_regs)

        critic_in = list(zip(obs, samp_acs))
        critic_rets = self.critic(critic_in, return_all_q=True)
        for a_i, probs, log_pi, pol_regs, (q, all_q) in zip(range(self.nagents), all_probs, all_log_pis, all_pol_regs, critic_rets):
            curr_agent = self.agents[a_i]
            v = (all_q * probs).sum(dim=1, keepdim=True)
            pol_target = q - v
            if soft:
                pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
            else:
                pol_loss = (log_pi * (-pol_target).detach()).mean()
            for reg in pol_regs:
                pol_loss += 1e-3 * reg  # 策略正则化
            disable_gradients(self.critic)
            pol_loss.backward()
            enable_gradients(self.critic)

            grad_norm = torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

            if logger is not None:
                logger.add_scalar('agent%i/losses/pol_loss' % a_i, pol_loss, self.niter)
                logger.add_scalar('agent%i/grad_norms/pi' % a_i, grad_norm, self.niter)

    def update_all_targets(self):
        """
        更新所有目标网络（在每个代理的正常更新后调用）
        """
        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)

    def prep_training(self, device='gpu'):
        """
        准备训练
        参数:
            device (str): 设备类型，'cpu'或'gpu'
        """
        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if self.pol_dev != device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if self.critic_dev != device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if self.trgt_pol_dev != device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if self.trgt_critic_dev != device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        """
        准备执行回合
        参数:
            device (str): 设备类型，'cpu'或'gpu'
        """
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if self.pol_dev != device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        保存所有智能体的训练参数到一个文件中
        参数:
            filename (str): 保存文件的路径
        """
        self.prep_training(device='cpu')  # 在保存前将参数移动到CPU
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents],
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()}}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                      reward_scale=10., pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4, **kwargs):
        """
        从多智能体环境实例化该类

        参数:
            env: 多智能体Gym环境
            gamma: 折扣因子
            tau: 目标网络的更新速率
            pi_lr: 策略网络的学习率
            q_lr: 评论家网络的学习率
            reward_scale: 奖励缩放
            pol_hidden_dim: 策略网络的隐藏层维度
            critic_hidden_dim: 评论家网络的隐藏层维度
            attend_heads: 注意力头的数量
        """
        agent_init_params = []
        sa_size = []
        for acsp, obsp in zip(env.action_space, env.observation_space):
            agent_init_params.append({'num_in_pol': obsp.shape[0], 'num_out_pol': acsp.n})
            sa_size.append((obsp.shape[0], acsp.n))

        init_dict = {'gamma': gamma, 'tau': tau, 'pi_lr': pi_lr, 'q_lr': q_lr, 'reward_scale': reward_scale,
                     'pol_hidden_dim': pol_hidden_dim, 'critic_hidden_dim': critic_hidden_dim, 'attend_heads': attend_heads,
                     'agent_init_params': agent_init_params, 'sa_size': sa_size}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        """
        从'保存'方法创建的文件中实例化该类
        参数:
            filename (str): 保存文件的路径
            load_critic (bool): 是否加载评论家网络参数
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)

        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
        return instance
