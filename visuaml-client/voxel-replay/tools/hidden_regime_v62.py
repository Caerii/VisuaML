"""Recurrent allocation across tasks sharing unobserved reliability."""
from argparse import ArgumentParser

import torch
from torch import nn

from adaptive_synthesis_v62 import OUT
from frozen_call_checkpoint import canonical

WIDTH = 24
TICKS = 12


class BeliefPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(6,WIDTH)
        self.attention = nn.MultiheadAttention(WIDTH,4,batch_first=True)
        self.norm = nn.LayerNorm(WIDTH)
        self.readout = nn.Linear(WIDTH,5)

    def project(self,current):
        return self.embed(current)

    def forward(self,current,history,missing):
        token = self.embed(current).unsqueeze(1)
        joined = torch.cat((history,token),1)
        pad = torch.cat((missing,torch.zeros_like(missing[:,:1])),1)
        value,_ = self.attention(token,joined,joined,key_padding_mask=pad,need_weights=False)
        memory = self.norm(value[:,0]+token[:,0])
        return self.readout(memory),memory


def features(gross,cost,budget,outcome,reuse):
    return torch.stack((gross/40,cost/20,budget/80,(outcome==1).float(),
                       (outcome==-1).float(),reuse.float()),1)


def train(seed,full_history):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    net = BeliefPolicy()
    optimizer = torch.optim.AdamW(net.parameters(),lr=.002)
    batch = 96
    best = []
    for update in range(980):
        regime = torch.rand(batch)>.5
        probability = torch.where(regime,.84,.16)
        budget = torch.randint(24,65,(batch,)).float()
        memory = torch.zeros(batch,TICKS,WIDTH)
        missing = torch.ones(batch,TICKS,dtype=torch.bool)
        last = torch.where(torch.rand(batch)<probability,1,-1)
        visits = []
        for t in range(TICKS):
            gain = torch.randint(9,38,(batch,)).float()
            cost = torch.randint(3,17,(batch,)).float()
            reuse = torch.rand(batch)<.1
            current = features(gain,cost,budget,last,reuse)
            logits,state = net(current,memory,missing)
            valid = torch.rand(batch)<probability
            can_prove = (cost<=budget) & ~reuse
            policy = torch.distributions.Bernoulli(logits=logits[:,0])
            action = policy.sample()*can_prove
            reward = (gain*valid-cost)*action
            budget = budget-cost*action
            # Cache hits are checked before this decision. Payoff from reuse
            # is action-independent, so omit it from optimization.
            visits.append((policy.log_prob(action),logits,reward/40,can_prove,
                           probability,cost/20,reuse.float(),action))
            last = torch.where(action.bool(),torch.where(valid,1,-1),0)
            if full_history:
                memory = torch.cat((memory[:,1:],net.embed(current).unsqueeze(1)),1)
                missing = torch.cat((missing[:,1:],torch.zeros_like(missing[:,:1])),1)
            else:
                memory = torch.cat((memory[:,:TICKS-1],state.unsqueeze(1)),1)
                missing = torch.cat((missing[:,:TICKS-1],torch.zeros_like(missing[:,:1])),1)
        total = torch.zeros(batch)
        loss = torch.zeros(())
        for logp,logits,reward,able,p,c,r,a in reversed(visits):
            total = total+reward
            advantage = total-logits[:,1]
            loss += ((-logp*advantage.detach())*able+.3*advantage.square()).mean()
            # Auxiliaries are forecasts, never authorization inputs.
            loss += .06*(nn.functional.binary_cross_entropy_with_logits(logits[:,2],p)
                         +(logits[:,3]-c).square().mean()
                         +nn.functional.binary_cross_entropy_with_logits(logits[:,4],r))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(),1.5)
        optimizer.step()
        if update%140==0:
            best.append(float(total.mean()))
    kind = "history" if full_history else "recurrent"
    # Trace the embedding used by the full-history control as a separate method.
    net.eval()
    traces = {"forward":(torch.zeros(1,6),torch.zeros(1,TICKS,WIDTH),
                         torch.ones(1,TICKS,dtype=torch.bool)),"project":(torch.zeros(1,6),)}
    torch.jit.freeze(torch.jit.trace_module(net,traces,check_trace=False),
                     preserved_attrs=["project"]).save(str(OUT/f"belief-{kind}-{seed}.pt"))
    return best


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--history",action="store_true")
    args = parser.parse_args()
    result = train(args.seed,args.history)
    kind = "history" if args.history else "recurrent"
    (OUT/f"training-{kind}-{args.seed}.json").write_bytes(canonical(dict(
        seed=args.seed,updates=980,trajectories_per_batch=96,kind=kind,reward_samples=result))+b"\n")
