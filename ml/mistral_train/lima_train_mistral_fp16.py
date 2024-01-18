import sys, torch, time, random, os, cv2, re, argparse, json, multiprocessing, itertools
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CURL_CA_BUNDLE'] = ''
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig
from sklearn.model_selection import KFold, train_test_split

device = "mps" if torch.backends.mps.is_available() else "cpu"

class args: # Uncomment for local run without argument
     dataset_dir = './dataset/'
     pytorch_device = 'cpu'
     val_proportion = 0.17
     time_patience = 3600*6
     patience = 100
     max_dur = 3600*24
     learning_rate = 0.001
     max_epochs = 1000
     local_files_only = False
     cache_dir = None

def parse_args():
   """ Parse commmand line arguments """
   
   parser = argparse.ArgumentParser(description='Train Transformer.')
   parser.add_argument("-a", "--dataset_dir", required=True,
                    metavar="/path/to/lima/dataset/",
                    help='Directory of the image and annotation dataset')
   parser.add_argument("-tp" , "--time_patience",
                          help="Early stopping's patience, in seconds rather than in number of epochs",
                          type=int, 
                          default=3600*3)
   parser.add_argument("-p" , "--patience",
                        help="Early stopping's patience",
                        type=int, 
                        default=40)
   parser.add_argument("-c" , "--cache_dir",
                      help="Cache dir for the model",
                      type=str, 
                      default='./outputs/')
   parser.add_argument("-l" , "--local_files_only",
                          help="Use locally downloaded vision model folder",
                          type=bool, 
                          default=False)
   parser.add_argument("-md" , "--max_dur",
                          help="Maximum training duration in seconds",
                          type=int, 
                          default=3600*24)
   parser.add_argument("-lr" , "--learning_rate",
                            help="Neural net's learning rate",
                            type=float, 
                            default=0.001)
   parser.add_argument("-me" , "--max_epochs",
                        help="Maximum number of epochs for training",
                        type=int, 
                        default=999999)
   parser.add_argument("-t" , "--val_proportion",
                    help="Proportion of dataset used for validation set."+\
                         "For example 0.4 will use 60% of train, 40% of val",
                    type=float, 
                    default=0.17)
   parser.add_argument("-d", "--pytorch_device", required=True,
                          help='PyTorch device, mps, cuda or cpu for example', 
                          default=device)
   
   args = parser.parse_args()
   return args

def loading_bar(extra_text, count, total, size=2):
    percent = float(count)/float(total)*100
    string = "\r" + str(int(count)).rjust(3,'0')+"/"+\
                    str(int(total)).rjust(3,'0') + ' [' + '='*int(percent/10)*\
                    size + ' '*(10-int(percent/10))*size + '] '
    sys.stdout.write(string + str(extra_text))

class Model(nn.Module):
    def __init__(self, local_files_only=False, cache_dir='./outputs/'):
        super(Model, self).__init__()
        trans_args = {'pretrained_model_name_or_path': "mistralai/Mistral-7B-v0.1", 
                      'cache_dir': cache_dir, 'local_files_only': local_files_only}
        self.trans = LlamaForCausalLM.from_pretrained(**trans_args)
        for name, param in self.trans.named_parameters():
            param.requires_grad = "norm" in name
            print((name, param.requires_grad))
        self.tok = AutoTokenizer.from_pretrained(**trans_args)
        self.adapation_prompt = nn.Parameter(torch.randn((200, 
                    self.trans.base_model.embed_tokens.embedding_dim))/100)
    
    def forward(self, toks):
        x = self.trans.base_model.embed_tokens(toks)
        x = torch.cat([self.adapation_prompt[None], x], dim=1)
        return self.trans(inputs_embeds=x, return_dict=True)['logits']

def conv_to_tokens(conv, tok):
    text = ''
    for i, answer in enumerate(conv):
        text += '\n-Asker: ' if i%2 == 0 else '\n-Expert: '
        text += answer
    
    text += '\n-Asker: ' if (i+1)%2 == 0 else '\n-Expert: '
    return tok.encode(text)[:1700] # Long conversations are cropped

def find_targets(toks, eot_id):
    targets = []
    robot_is_talking = False
    for i, tok in enumerate(toks):
        if toks[i-4:i] == [28733, 966, 2482, 28747]: # Manual
            robot_is_talking = False
        if toks[i-5:i] == [28733, 28741, 1252, 263, 28747]: # Manual
            robot_is_talking = True
        if robot_is_talking:
            targets.append(i)
    return targets

class args: # Uncomment for local run without argument
     dataset_dir = './dataset/'
     pytorch_device = 'cpu'
     val_proportion = 0.17
     time_patience = 3600*6
     patience = 100
     max_dur = 3600*24
     learning_rate = 0.001
     max_epochs = 1000
     local_files_only = False
     cache_dir = None


if __name__ == "__main__" :
    np.random.seed(1912)
    random.seed(1912)
    torch.manual_seed(1912)
    
    args = parse_args()
    print(args)
    dataset_dir = args.dataset_dir
    print(dataset_dir)
    device = args.pytorch_device  
    
    train = pd.read_json(os.path.join(args.dataset_dir, 'public_datasets', 
                                       'lima','train.jsonl'), lines=True)
    
    #scaler = torch.cuda.amp.GradScaler() # This should not work for MPS
    model = Model(local_files_only=args.local_files_only, cache_dir=args.cache_dir)
    model = model.to(device)
    opt = torch.optim.NAdam(model.parameters(), lr=args.learning_rate)
    print(('nb_params', sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
    train['tokens'] = train['conversations'].map(lambda x: conv_to_tokens(x, model.tok))
    train['target_tokens'] = train['tokens'].map(lambda x: find_targets(x, model.tok.eos_token_id))
    train, val = train_test_split(train, test_size=int(len(train)*args.val_proportion), 
                                  random_state=2)
    train.to_csv('./outputs/train.csv') # Check which sentences were chosen or dont use val
    
    # Trainer
    train_info = {'waited' : 0, 'patience' : args.patience, 'epoch' : 0, 'duration': 0.0,
                  'start_time': time.time(), 'max_dur' : args.max_dur, 'waited': 0,
                  'max_epochs': args.max_epochs, 'batch_size': 1, 
                  'time_patience': args.time_patience, 
                  'min_validated': np.inf, 'time_last_best': time.time(),
                  'history': pd.DataFrame(columns=['va_loss'])}
    print(train_info)
    nsplits = len(train) // train_info['batch_size']
    kf_val = KFold(n_splits=len(val) // train_info['batch_size'], shuffle=False)
    np.random.seed(1912)
    while((train_info['duration'] < train_info['max_dur'])&
          (train_info['waited'] < train_info['patience']) & 
          (time.time() - train_info['time_last_best'] < train_info['time_patience']) & 
          (train_info['epoch'] < train_info['max_epochs'])):
        train_info['waited'] += 1
        train_info['epoch'] += 1
        
        kf = KFold(n_splits=nsplits, shuffle=True)
        model = model.train(mode=True)
        for batch_nb, (_, batch_indexes) in enumerate(kf.split(train)):
            batch = train.iloc[batch_indexes]
            tokens = torch.from_numpy(np.array(batch['tokens'].values[0]))[None, :].to(device)
            
            #with torch.cuda.amp.autocast(): # This should not work for MPS
            pred = model(tokens[:, :-1])[:, 200:].permute((0, 2, 1))
            loss = torch.nn.functional.cross_entropy(pred, tokens[:, 1:].to(torch.long), 
                                                         reduction='none')
                
            # Keep only robot's answers
            indexes = torch.from_numpy(np.array(batch['target_tokens'].iloc[0])).to(device)
            indexes = indexes - 1
            loss = torch.index_select(loss, 1, indexes).mean()
        
            opt.zero_grad()
            opt.zero_grad()
            loss.backward()
            opt.step()
            #scaler.scale(loss).backward() # This should not work for MPS
            #scaler.step(opt)
            #scaler.update()
            
            current_losses = {'epoch': train_info['epoch'], 't_loss': round(loss.item(), 10)}
            train_info['history'] = pd.concat([train_info['history'],  
                                               pd.DataFrame([current_losses])], axis=0)
            
            if (batch_nb + 1) % (nsplits // 10 + 1) == 0:
                loading_bar(f'{current_losses}', batch_nb+1, nsplits)
        
        losses = []
        model = model.train(mode=False)
        for batch_nb, (_, batch_indexes) in enumerate(kf_val.split(val)):
            batch = val.iloc[batch_indexes]
            tokens = torch.from_numpy(np.array(batch['tokens'].values[0]))[None, :].to(device)
            with torch.no_grad():
                #with torch.cuda.amp.autocast():
                pred = model(tokens[:, :-1])[:, 200:].permute((0, 2, 1))
                loss = torch.nn.functional.cross_entropy(pred, tokens[:, 1:].to(torch.long), 
                                                            reduction='none')
                
                # Keep only robot's answers
                indexes = torch.from_numpy(np.array(batch['target_tokens'].iloc[0])).to(device)
                indexes = indexes - 1
                loss = torch.index_select(loss, 1, indexes).mean()
                
                losses.append(loss.item())
        
        current_losses['va_loss'] = round(np.mean(losses), 10)
        train_info['history'].iloc[-1, train_info['history'].columns.get_loc('va_loss')
                     ] = current_losses['va_loss']
        
        current_losses['t_loss'] = round(train_info['history'][-nsplits:]['t_loss'
                                                      ].mean(), 10)
        
        train_info['duration'] = time.time() - train_info['start_time']
        loading_bar(f"{current_losses} duration {round(train_info['duration'], 2)}\n", 
                    nsplits, nsplits)
        print({x: y for x,y in train_info.items() if x!='history'})
        
        if current_losses['va_loss'] < train_info['min_validated']: # Best NN so far
             train_info['min_validated'] = current_losses['va_loss']
             train_info['waited'], train_info['time_last_best']  = 0, time.time()
             torch.save(model.state_dict(), './outputs/model_weights.pt')
    
    print('Training successfully runned.')
    