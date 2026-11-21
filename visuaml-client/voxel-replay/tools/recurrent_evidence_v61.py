"""Train a shared-weight evidence policy from complete budgeted rollouts."""
import torch
from torch import nn

from reusable_evidence_v59 import OUT as V59

OUT = V59.parent/"recurrent-evidence-v61"
WIDTH = 48


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(11,WIDTH)
        self.attention = nn.MultiheadAttention(WIDTH,4,batch_first=True)
        self.norm = nn.LayerNorm(WIDTH)
        self.block = nn.Sequential(nn.Linear(WIDTH,WIDTH),nn.GELU(),nn.Linear(WIDTH,WIDTH))
        self.queries = nn.Linear(WIDTH,2)
        self.terminal = nn.Linear(WIDTH,2)

    def forward(self, features, memory):
        tokens = torch.cat((self.embed(features),memory.unsqueeze(1)),1)
        attention,_ = self.attention(tokens,tokens,tokens,need_weights=False)
        tokens = self.norm(tokens+attention)
        tokens = tokens+self.block(tokens)
        recurrent = tokens[:,3]
        terminal = self.terminal(recurrent)
        return torch.cat((terminal[:,:1],self.queries(tokens[:,:3]).flatten(1)),1),terminal[:,1],recurrent


def encode(costs, payoff, priors, status, budget):
    cap = budget[:,None].expand_as(payoff)
    return torch.cat((costs/100, payoff.unsqueeze(-1)/100,priors,
        nn.functional.one_hot(status,4).float(),cap.unsqueeze(-1)/100,
        (costs[:,:,1]<=cap).float().unsqueeze(-1)),2)


def mask(costs,status,budget):
    available = costs<=budget[:,None,None]
    queries = torch.stack((available[:,:,0] & (status==0),
                           available[:,:,1] & (status<2)),2)
    return torch.cat((torch.ones_like(budget[:,None],dtype=torch.bool),queries.flatten(1)),1)


def train():
    torch.manual_seed(8161)
    torch.set_num_threads(1)
    model = Policy()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0015)
    batch = 96
    ids = torch.arange(batch)
    returns = []
    for update in range(1700):
        costs = torch.stack((torch.randint(1,13,(batch,3)),torch.randint(5,53,(batch,3))),2).float()
        payoff = torch.randint(8,120,(batch,3)).float()
        p = torch.rand(batch,3)*.7+.12
        r = torch.rand(batch,3)*(1-p)
        priors = torch.stack((p,r),2)
        # One unobserved regime is shared by all candidates in a rollout.
        shift = torch.randint(-2,3,(batch,1))*.11
        truth = torch.minimum((p+shift).clamp_min(.02),1-r)
        draws = torch.rand(batch,3)
        status = torch.where(torch.rand(batch,3)<.08,torch.where(draws<truth,3,2),0)
        budget = torch.randint(8,104,(batch,)).float()
        memory = torch.zeros(batch,WIDTH)
        live = torch.ones(batch,dtype=torch.bool)
        trajectory = []
        for _ in range(7):
            logits,value,memory = model(encode(costs,payoff,priors,status,budget),memory)
            distribution = torch.distributions.Categorical(logits=logits.masked_fill(~mask(costs,status,budget),-1e9))
            action = distribution.sample()
            active = live & (action!=0)
            index = (action-1).clamp_min(0)//2
            strong = (action%2==0).long()
            paid = costs[ids,index,strong]*active
            prior_best = (payoff*(status==3)).amax(1)
            valid = draws[ids,index]<truth[ids,index]
            detected = (draws[ids,index]>=truth[ids,index]) & (draws[ids,index]<truth[ids,index]+r[ids,index])
            outcome = torch.where(strong.bool(),torch.where(valid,3,2),torch.where(detected,2,1))
            status = status.clone()
            status[ids,index] = torch.where(active,outcome,status[ids,index])
            budget = budget-paid
            reward = ((payoff*(status==3)).amax(1)-prior_best-paid)/100
            trajectory.append((distribution.log_prob(action),value,distribution.entropy(),reward,live))
            live = active
        total = torch.zeros(batch)
        loss = torch.zeros(())
        for logp,value,entropy,reward,active in reversed(trajectory):
            total = total+reward
            advantage = total-value
            loss = loss+((-logp*advantage.detach()+.4*advantage.square()-.008*entropy)*active).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
        if update%100==0:
            returns.append(float(total.mean()))
    model.eval()
    traced = torch.jit.trace(model,(torch.zeros(1,3,11),torch.zeros(1,WIDTH)),check_trace=False)
    torch.jit.freeze(traced).save(str(OUT/"policy.pt"))
    import json
    (OUT/"training.json").write_text(json.dumps(dict(seed=8161,updates=1700,batch_size=batch,
           reward_samples=returns,teacher="none",objective="terminal certified payoff less spent charges"),
           indent=2)+"\n")


if __name__=="__main__":
    train()
