"""Shared exact stopping contract and learned query proposals."""
import torch
from torch import nn


@torch.jit.script
def certify(lower: torch.Tensor, upper: torch.Tensor, failure: int) -> int:
    # Costs are integers; action index is the final tie breaker.
    assert lower.numel() == upper.numel()
    assert bool(((lower >= 0) & (upper <= failure) & (lower <= upper)).all())
    count = lower.numel()
    if bool((lower == failure).all()):
        return count
    for i in range(count):
        if int(upper[i]) == failure:
            continue
        best = True
        for j in range(count):
            if i != j and int(upper[i])*count+i >= int(lower[j])*count+j:
                best = False
        if best:
            return i
    return -1


class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(10,48),nn.ReLU(),nn.Linear(48,48),
                                     nn.ReLU(),nn.Linear(48,9))

    def forward(self, domain: int, lower: torch.Tensor, upper: torch.Tensor,
                queried: torch.Tensor, budget: int, failure: int) -> int:
        features = torch.cat((lower.float()/failure, upper.float()/failure, queried.float(),
                              torch.tensor([float(budget)])))
        return int(self.network(features).argmax())


class CheckedInference(nn.Module):
    def __init__(self, controller, forecast):
        super().__init__()
        self.controller = controller
        self.forecast = forecast

    def forward(self, domain: int, walls: torch.Tensor, signal: torch.Tensor,
                edits: torch.Tensor, target: int, horizon: int, budget: int,
                costly_edits: bool, source_slots: torch.Tensor, plan_slots: torch.Tensor,
                live_locals: torch.Tensor, policy: int = 0, coverage: int = 0):
        # Proposal IDs 0–2 query an option, 3–6 claim an action or Neither,
        # 7 abstains, and 8 requests a targeted compiler replay.
        count = 3 if domain == 0 else 2
        failure = 32 if domain == 0 else 4
        lower = torch.zeros(3,dtype=torch.long)
        upper = torch.full((3,),failure,dtype=torch.long)
        queried = torch.zeros(3,dtype=torch.long)
        assert 0 <= budget <= 2
        if domain == 0:
            assert 1 <= horizon <= 8 and 0 <= target < walls.numel()
            start_positions = torch.nonzero(signal.flatten())
            assert start_positions.numel()==1
            start = int(start_positions[0,0])
            assert start != target
            width = walls.size(3)
            distance = abs(start//width-target//width)+abs(start%width-target%width)
            for action in range(3):
                penalty = 9 if costly_edits and action != 0 else 0
                lower[action] = distance+penalty if distance <= horizon else failure
        else:
            # Reference is legal by definition. Candidate rank is an explicit
            # input contract for this witness: one fused transition vs two.
            lower[0],upper[0],lower[1] = 2,2,1
            assert source_slots.numel()==2 and plan_slots.numel()==2
            assert bool(((source_slots>=0)&(source_slots<live_locals.numel())).all())
            assert bool(((plan_slots>=0)&(plan_slots<live_locals.numel())).all())

        journal = torch.full((3,11),-1,dtype=torch.long)
        frames = torch.zeros((2,8,walls.size(2),walls.size(3)))
        for turn in range(3):
            remaining = budget-turn
            verdict = certify(lower[:count],upper[:count],failure)
            if policy == 0:
                proposal = self.controller(domain,lower,upper,queried,remaining,failure)
            else:
                proposal = 7
                if verdict >= 0:
                    proposal = 3+verdict
                elif remaining > 0:
                    for i in range(count):
                        if not int(queried[i]) and int(lower[i]) != int(upper[i]):
                            proposal = i
                            break
            journal[turn,0]=proposal
            journal[turn,1]=verdict
            journal[turn,2:5]=lower
            journal[turn,5:8]=upper
            if 3 <= proposal <= 7:
                claim = proposal-3 if proposal != 7 else -1
                return journal,frames,claim if claim==verdict else -1,claim==verdict
            candidate_replay = domain == 1 and proposal == 8
            if remaining <= 0 or (not candidate_replay and not 0 <= proposal < count):
                break
            if domain == 0:
                if queried[proposal] != 0 or coverage < horizon:
                    break
                terrain = walls.clone()
                if proposal != 0:
                    edit = int(edits[proposal-1])
                    assert 0 <= edit < terrain.numel()
                    terrain.flatten()[edit] = 0
                _, history, _ = self.forecast.forecast(terrain,signal,0,target,0,horizon)
                frames[turn,:horizon] = history[:,0]
                arrival = 0
                for tick in range(horizon):
                    if int(history[tick].flatten()[target]):
                        arrival = tick+1
                        break
                cost = failure if arrival==0 else arrival+(9 if costly_edits and proposal!=0 else 0)
                lower[proposal],upper[proposal],queried[proposal] = cost,cost,1
                journal[turn,8]=arrival
                journal[turn,9]=horizon
            else:
                # Model of the existing ordered local.get pair. Both original
                # boundaries consume one fuel and push one selected i32 each.
                if candidate_replay and queried[2]==0:
                    mismatch = live_locals[source_slots] != live_locals[plan_slots]
                    queried[2] = 1
                    difference = 1 if bool(mismatch[0]) else 2 if bool(mismatch[1]) else 0
                    journal[turn,8]=difference
                    if difference:
                        lower[1],upper[1]=failure,failure
                elif proposal==1 and queried[1]==0:
                    legal = bool((source_slots==plan_slots).all())
                    queried[1]=1
                    lower[1],upper[1] = (1,1) if legal else (failure,failure)
                    journal[turn,8] = int(legal)
                else:
                    break
        return journal,frames,-1,False


def train():
    from random import Random
    from pocket_world_inference import OUT as WORLD
    from composed_inference import walk

    torch.set_num_threads(1)
    torch.manual_seed(508)
    rng = Random(508)
    features,labels = [], []

    def add(lower,upper,queried,budget,failure,proposal):
        features.append([v/failure for v in lower+upper]+queried+[budget])
        labels.append(proposal)

    for _ in range(8500):
        deadline = rng.randint(3,8)
        distance = rng.randint(1,deadline)
        penalty = [0]+[rng.choice((0,9))]*2
        costs = [rng.choice(list(range(distance+p,deadline+p+1))+[32]) for p in penalty]
        lower = [distance+p for p in penalty]
        upper = [32]*3
        queried = [0]*3
        for budget in range(2,-1,-1):
            winner = certify(torch.tensor(lower),torch.tensor(upper),32)
            choices = sorted((i for i in range(3) if not queried[i]),key=lambda i:(lower[i],i))
            proposal = 3+winner if winner >= 0 else choices[0] if budget else 7
            add(lower,upper,queried,budget,32,proposal)
            if proposal >= 3:
                break
            if rng.random() < .45:
                proposal = rng.choice(choices)
            lower[proposal]=upper[proposal]=costs[proposal]
            queried[proposal]=1

    for _ in range(700):
        for legal in (False,True):
            for budget in (1,2):
                add([2,1,0],[2,4,4],[0,0,0],budget,4,8 if budget==2 else 1)
                add([2,1,0],[2,4,4],[0,0,1],1,4,1)
                add([2,4,0],[2,4,4],[0,0,1],1,4,3)
                c = 1 if legal else 4
                add([2,c,0],[2,c,4],[0,1,int(budget==2)],0,4,4 if legal else 3)

    x,y = torch.tensor(features),torch.tensor(labels)
    controller = Controller()
    optimizer = torch.optim.AdamW(controller.parameters(),lr=.003)
    for _ in range(1500):
        batch = torch.randint(len(y),(128,))
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(controller.network(x[batch]),y[batch])
        loss.backward()
        optimizer.step()
    simulator = torch.jit.load(str(WORLD/"inference.pt"))
    graph = torch.jit.freeze(torch.jit.script(CheckedInference(controller,simulator).eval()))
    assert not {"prim::PythonOp","prim::CallMethod"} & {node.kind() for node in walk(graph.inlined_graph)}
    out = WORLD.parent/"checked-decisions"
    graph.save(str(out/"inference.pt"))
    (out/"inference.ir").write_text(str(graph.inlined_graph).rstrip()+"\n")


if __name__=="__main__":
    train()
