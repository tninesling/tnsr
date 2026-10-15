"""Regenerate the small committed PPO oracle using forty-five's original updater."""
import argparse
import copy
import json
from pathlib import Path
import sys
from export import load_archive, export_model, import_project, model_config
from validate_ppo import fixture, anchor_outputs, reference_step


def main():
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forty-five-root',type=Path,required=True)
    parser.add_argument('--separate',action='store_true')
    args=parser.parse_args()
    sys.path.insert(0,str(args.forty_five_root/'python'))
    torch.set_num_threads(2)
    directory=Path('examples/forty_five/fixtures/tiny')
    model=load_archive(directory,args.forty_five_root)
    if args.separate:
        PolicyValueNet,_=import_project(args.forty_five_root)
        config=model_config(model);config['use_separate_critic']=True
        torch.manual_seed(23)
        separate=PolicyValueNet(action_dim=46,**config)
        separate.load_state_dict(model.state_dict(),strict=False)
        model=separate.eval()
        directory=directory.with_name('tiny-separate')
        export_model(model,directory)
    original,native=fixture(model,args.forty_five_root,True)
    anchor=copy.deepcopy(model)
    with torch.no_grad():anchor.bid_head.bias.add_(torch.linspace(-.2,.2,5))
    native['anchor']=anchor_outputs(anchor,native)
    settings=dict(learning_rate=5e-5,value_learning_rate=2e-4,entropy_coef=.005,
                  anchor_kl_coef=.03,microbatch=3,scheduler_updates=4)
    optimizer=torch.optim.Adam([dict(params=model.policy_parameters(),lr=5e-5),dict(params=model.value_parameters(),lr=2e-4)])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=4,eta_min=1e-6)
    steps=[]
    for _ in range(2):
        metrics,gradients=reference_step(model,anchor,original,optimizer,3,.005,.03)
        scheduler.step()
        steps.append(dict(metrics=metrics,gradients={k:v.flatten().tolist() for k,v in gradients.items()},
                          weights={k:v.detach().flatten().tolist() for k,v in model.named_parameters()},
                          learning_rates=scheduler.get_last_lr()))
    output=dict(torch_version=torch.__version__,batch=native,settings=settings,steps=steps)
    (directory/'ppo.json').write_text(json.dumps(output,separators=(',',':'))+'\n')
if __name__=='__main__':main()
