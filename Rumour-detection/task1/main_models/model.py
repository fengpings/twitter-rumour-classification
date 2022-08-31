import torch
import torch.nn as nn
from transformers import BertModel
from transformers import AutoModel
import transformers
transformers.logging.set_verbosity_error()
class TextBertClassifier(nn.Module):

    def __init__(self):
        super(TextBertClassifier, self).__init__()
        #Instantiating BERT model object
        
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert_layer.parameters():
            param.requires_grad = True
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.ffnn = nn.Sequential(nn.Linear(768,128),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                 nn.Linear(128,64),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                  nn.Linear(64,1),
                                  nn.Sigmoid()
                                 )

    def forward(self, seq_, attn_masks, seg_, stats):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        outputs = self.bert_layer(seq_, attention_mask = attn_masks,token_type_ids = seg_, return_dict=True)
        cont_reps = outputs.last_hidden_state

        #Obtaining the representation of [CLS] head (the first token)
        cls_rep = cont_reps[:, 0]

        # x = torch.cat((cls_rep, stats),dim=1)
        #Feeding cls_rep to the classifier layer
        logits = self.ffnn(cls_rep)

        return logits
class TextBertTFClassifier(nn.Module):

    def __init__(self):
        super(TextBertTFClassifier, self).__init__()
        #Instantiating BERT model object
        
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert_layer.parameters():
            param.requires_grad = True
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.ffnn = nn.Sequential(nn.Linear(811,128),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                 nn.Linear(128,64),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                  nn.Linear(64,1),
                                  nn.Sigmoid()
                                 )

    def forward(self, seq_, attn_masks, seg_, stats):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        outputs = self.bert_layer(seq_, attention_mask = attn_masks,token_type_ids = seg_, return_dict=True)
        cont_reps = outputs.last_hidden_state

        #Obtaining the representation of [CLS] head (the first token)
        cls_rep = cont_reps[:, 0]

        x = torch.cat((cls_rep, stats),dim=1)
        #Feeding cls_rep to the classifier layer
        logits = self.ffnn(x)

        return logits
class TextBertweetTFClassifier(nn.Module):

    def __init__(self):
        super(TextBertweetTFClassifier, self).__init__()
        #Instantiating BERT model object
        
        self.bert_layer = AutoModel.from_pretrained("vinai/bertweet-base", cache_dir='./bertweet_base_model',local_files_only=True)
#         for param in self.bert_layer.parameters():
#             param.requires_grad = True
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.ffnn = nn.Sequential(nn.Linear(811,128),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                 nn.Linear(128,64),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                  nn.Linear(64,1),
                                  nn.Sigmoid()
                                 )

    def forward(self, seq_, attn_masks, seg, stats):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        outputs = self.bert_layer(seq_, attention_mask = attn_masks, return_dict=True)
        cont_reps = outputs.last_hidden_state

        #Obtaining the representation of [CLS] head (the first token)
        cls_rep = cont_reps[:, 0]

        x = torch.cat((cls_rep, stats), dim=1)
        #Feeding cls_rep to the classifier layer
        logits = self.ffnn(x)

        return logits
class TextBertweetClassifier(nn.Module):

    def __init__(self):
        super(TextBertweetClassifier, self).__init__()
        #Instantiating BERT model object
        
        self.bert_layer = AutoModel.from_pretrained("vinai/bertweet-base", cache_dir='./bertweet_base_model',local_files_only=True)
#         for param in self.bert_layer.parameters():
#             param.requires_grad = True
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.ffnn = nn.Sequential(nn.Linear(768,128),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                 nn.Linear(128,64),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                  nn.Linear(64,1),
                                  nn.Sigmoid()
                                 )

    def forward(self, seq_, attn_masks,seg,  stats):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        outputs = self.bert_layer(seq_, attention_mask = attn_masks, return_dict=True)
        cont_reps = outputs.last_hidden_state

        #Obtaining the representation of [CLS] head (the first token)
        cls_rep = cont_reps[:, 0]

#         x = torch.cat((cls_rep, stats), dim=1)
        #Feeding cls_rep to the classifier layer
        logits = self.ffnn(cls_rep)

        return logits
