import torch
from torch import nn
import torch.nn.functional as F
from .models import A2V_Attention, V2A_Attention
from .models import EncoderLayer, Encoder, DecoderLayer, Decoder
from torch.nn import MultiheadAttention

class RNNEncoder(nn.Module):
    def __init__(self, audio_dim, visual_dim, d_model, num_layers):
        super(RNNEncoder, self).__init__()

        self.d_model = d_model
        # self.audio_rnn = nn.LSTM(audio_dim, int(d_model / 2), num_layers=num_layers, batch_first=True,
        #                          bidirectional=True, dropout=0.2)
        self.audio_rnn = nn.LSTM(audio_dim, d_model, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.2)
        self.visual_rnn = nn.LSTM(visual_dim, d_model, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.2)

    def forward(self, audio_feature, visual_feature):
        audio_output, _ = self.audio_rnn(audio_feature)
        visual_output, _ = self.visual_rnn(visual_feature)
        return audio_output, visual_output


class InternalTemporalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(InternalTemporalRelationModule, self).__init__()
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        self.encoder = Encoder(self.encoder_layer, num_layers=2)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)
        # add relu here?

    def forward(self, feature):
        # feature: [seq_len, batch, dim]
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)

        return feature


class CrossModalRelationAttModule(nn.Module):
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(CrossModalRelationAttModule, self).__init__()

        self.decoder_layer = DecoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        self.decoder = Decoder(self.decoder_layer, num_layers=1)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feature, memory_feature):
        query_feature = self.affine_matrix(query_feature)
        output = self.decoder(query_feature, memory_feature)

        return output


class CAS_Module(nn.Module):
    def __init__(self, d_model, num_class):
        super(CAS_Module, self).__init__()
        self.classifier = nn.Conv1d(in_channels=d_model, out_channels=num_class, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, content):
        content = content.permute(0, 2, 1)
        out = self.classifier(content)
        out = out.permute(0, 2, 1)
        return out


class Multimodal_Gate(nn.Module):
    def __init__(self, input_dim):
        super(Multimodal_Gate, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim//2)
        self.fc2 = nn.Linear(input_dim//2, 1)

    def forward(self, audio, visual):
        # Concatenate audio and visual features along the last dimension
        combined_features = torch.cat((audio, visual), dim=-1)
    
        # Apply fully connected layers
        x = F.relu(self.fc1(combined_features))
        gate_weights = torch.sigmoid(self.fc2(x))

        return gate_weights

class Unimodal_Gate(nn.Module):
    def __init__(self, input_dim):
        super(Unimodal_Gate, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        gate_weights = torch.sigmoid(self.fc(x))
        return gate_weights

class SupvLocalizeModule(nn.Module):
    def __init__(self, d_model):
        super(SupvLocalizeModule, self).__init__()
        # self.affine_concat = nn.Linear(2*256, 256)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(d_model, 1)  # start and end
        self.event_classifier = nn.Linear(d_model, 28)
        # self.cas_model = CAS_Module(d_model, num_class=28)

    def forward(self, fused_content):
        max_fused_content, _ = fused_content.transpose(1, 0).max(1)
        logits = self.classifier(fused_content)
        # scores = self.softmax(logits)
        class_logits = self.event_classifier(max_fused_content)
        # class_logits = self.event_classifier(fused_content.transpose(1,0))
        # sorted_scores_base,_ = class_logits.sort(descending=True, dim=1)
        # topk_scores_base = sorted_scores_base[:, :4, :]
        # class_logits = torch.mean(topk_scores_base, dim=1)
        class_scores = class_logits

        return logits, class_scores

class WeaklyLocalizationModule(nn.Module):
    def __init__(self, input_dim):
        super(WeaklyLocalizationModule, self).__init__()

        self.hidden_dim = input_dim  # need to equal d_model
        self.classifier = nn.Linear(self.hidden_dim, 1)  # start and end
        self.event_classifier = nn.Linear(self.hidden_dim, 29)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fused_content):
        fused_content = fused_content.transpose(0, 1)
        max_fused_content, _ = fused_content.max(1)
        # confident scores
        is_event_scores = self.classifier(fused_content)
        # classification scores
        raw_logits = self.event_classifier(max_fused_content)[:, None, :]
        # fused
        fused_logits = is_event_scores.sigmoid() * raw_logits
        # Training: max pooling for adapting labels
        logits, _ = torch.max(fused_logits, dim=1)
        event_scores = self.softmax(logits)

        return is_event_scores.squeeze(), raw_logits.squeeze(), event_scores

class AudiovisualInter(nn.Module):
    def __init__(self, d_model, n_head, head_dropout=0.1):
        super(AudiovisualInter, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.visual_multihead = MultiheadAttention(d_model, num_heads=n_head, dropout=head_dropout)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, visual_feat, audio_feat):
        # visual_feat, audio_feat: [10, batch, 256]
        global_feat = visual_feat * audio_feat
        memory = torch.cat([audio_feat, visual_feat], dim=0)
        mid_out = self.visual_multihead(global_feat, memory, memory)[0]
        output = self.norm1(global_feat + self.dropout(mid_out))

        return output
    
class AudioVisualInteractionGraph(nn.Module):
    def __init__(self, visual_dim, audio_dim, num_neighbors=4):
        super(AudioVisualInteractionGraph, self).__init__()
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.num_neighbors = num_neighbors
        
        # Learnable parameters
        self.visual_weights = torch.nn.Parameter(torch.rand(visual_dim, visual_dim))
        self.audio_weights = torch.nn.Parameter(torch.rand(audio_dim, audio_dim))
        
    def forward(self, visual_features, audio_features):
        # Compute Mahalanobis distance between visual and audio features
        visual_mahalanobis = torch.matmul(visual_features, self.visual_weights)
        audio_mahalanobis = torch.matmul(audio_features, self.audio_weights)
        
        # Compute interaction graph based on Mahalanobis distance
        interaction_graph = torch.exp(-torch.norm(visual_mahalanobis.unsqueeze(2) - audio_mahalanobis.unsqueeze(1), dim=-1))
        
        # Find top k neighbors for each feature
        _, top_indices = torch.topk(interaction_graph, self.num_neighbors, dim=1)
        
        # Gather neighboring features for each sample
        visual_neighbors = torch.gather(visual_features, 1, top_indices.unsqueeze(3).expand(-1, -1, -1, self.visual_dim))
        audio_neighbors = torch.gather(audio_features, 1, top_indices.unsqueeze(3).expand(-1, -1, -1, self.audio_dim))
        
        # Compute enhanced features by averaging neighbors
        enhanced_visual_features = torch.mean(visual_neighbors, dim=2)
        enhanced_audio_features = torch.mean(audio_neighbors, dim=2)
        
        return enhanced_visual_features, enhanced_audio_features

class weak_main_model(nn.Module):
    def __init__(self, config):
        super(weak_main_model, self).__init__()
        self.config = config
        self.verbose = False # self.config["verbose"]
        self.alpha = self.config["alpha"]
        self.n_heads = self.config["n_heads"]
        self.visual_input_dim = self.config["visual_inputdim"]
        self.audio_input_dim = self.config["audio_inputdim"]

        self.visual_fc_dim = 512
        self.d_model = self.config["d_model"]

        # self.v_fc = nn.Linear(self.visual_input_dim, self.visual_fc_dim)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.2)

        ### ---> multi-head self attention
        self.visual_self_attention = MultiheadAttention(
            embed_dim=self.visual_input_dim, 
            num_heads=self.n_heads, 
            dropout=0.1)
        self.audio_self_attention = MultiheadAttention(
            embed_dim=self.audio_input_dim, 
            num_heads=self.n_heads, 
            dropout=0.1)
        self.self_att_visual_norm = nn.LayerNorm(self.visual_input_dim)
        self.self_att_audio_norm = nn.LayerNorm(self.audio_input_dim)
        self.self_att_visual_mlp = nn.Sequential(
            nn.Linear(self.visual_input_dim, self.visual_input_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.self_att_audio_mlp = nn.Sequential(
            nn.Linear(self.audio_input_dim, self.audio_input_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        ### <---

        ### ---> spatial-channel attention
        self.visual_spatial_channel_att = A2V_Attention()
        self.audio_spatial_channel_att = V2A_Attention()
        ### <---

        ### ---> relation module
        self.audio_visual_rnn_layer = RNNEncoder(audio_dim=self.visual_input_dim, visual_dim=self.visual_input_dim, d_model=self.d_model, num_layers=1)
        
        self.visual_encoder = InternalTemporalRelationModule(input_dim=self.visual_input_dim, d_model=self.d_model, feedforward_dim=2048)
        self.visual_decoder = CrossModalRelationAttModule(input_dim=self.visual_input_dim, d_model=self.d_model, feedforward_dim=1024)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=512, d_model=self.d_model, feedforward_dim=2048)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=512, d_model=self.d_model, feedforward_dim=1024)

        self.audio_gated = Unimodal_Gate(self.d_model)
        self.visual_gated = Unimodal_Gate(self.d_model)
        self.global_av_gate = Multimodal_Gate(self.d_model)
        self.global_gate_cas = nn.Linear(self.d_model, 29)
        # self.global_gate_cas = CAS_Module(d_model=self.d_model, num_class=29)

        self.AVInter = AudiovisualInter(self.d_model, n_head=4, head_dropout=0.2)
        self.VAInter = AudiovisualInter(self.d_model, n_head=4, head_dropout=0.2)
        
        self.visual_norm = nn.LayerNorm(self.d_model)
        self.audio_norm = nn.LayerNorm(self.d_model)

        self.audio_cas = nn.Linear(self.d_model, 29)
        self.visual_cas = nn.Linear(self.d_model, 29)
        self.localize_module = WeaklyLocalizationModule(self.d_model)

    def forward(self, visual_feature, audio_feature):
        if self.verbose: print('@visual feature: ', visual_feature.shape) # [b, 10, 7, 7, 512]
        if self.verbose: print('@audio feature: ', audio_feature.shape) # [b, 10, 128]

        # ============================== Adaptive Features (Optional) ===============================
        # optional, make the model adaptive to different visual features (e.g., VGG ,ResNet)
        # visual_feature = self.dropout(self.relu(self.v_fc(visual_feature)))
        # if self.verbose: print('@visual feature: ', visual_feature.shape) # [b, 10, 7, 7, 512]

        # ============================== Self Attention (Visual) ====================================
        b, t, w, h, c = visual_feature.shape
        visual_feature = visual_feature.view(b*t, -1, c) # [b*t, w*h, 512]
        visual_feature = self.self_att_visual_norm(visual_feature)
        visual_attention, _ = self.visual_self_attention(visual_feature, visual_feature, visual_feature) # [b*t, w*h, 512]
        if self.verbose: print('@self attention (visual): ', visual_attention.shape)
        enhanced_visual_feature = self.self_att_visual_norm(visual_feature + visual_attention) # [b*t, h*w, 512]
        enhanced_visual_feature = self.self_att_visual_mlp(enhanced_visual_feature)
        enhanced_visual_feature = enhanced_visual_feature.view(b, t, h, w, c) # [b, t, h, w, 512]
        if self.verbose: print('@enhanced_visual_feature: ', enhanced_visual_feature.shape) # [b, t, h, w, 512]

        # ============================== self Attention (Audio) ======================================
        b, t, c = audio_feature.shape
        audio_feature = audio_feature.view(b, t, -1) # [b, t, 128]
        audio_feature = self.self_att_audio_norm(audio_feature)
        audio_attention, _ = self.audio_self_attention(audio_feature, audio_feature, audio_feature) # [b, t, 128]
        if self.verbose: print('@self attention (audio): ', audio_attention.shape)
        enhanced_audio_feature = self.self_att_audio_norm(audio_feature + audio_attention)
        enhanced_audio_feature = self.self_att_audio_mlp(enhanced_audio_feature)
        if self.verbose: print('@enhanced_audio_feature: ', enhanced_audio_feature.shape) # [b, t, 128]

        # ============================== Spatial Channel Attention ====================================
        c_s_visual_feat = self.visual_spatial_channel_att(enhanced_visual_feature, enhanced_audio_feature)
        if self.verbose: print('@visual spatial-channel attention: ', c_s_visual_feat.shape) # [batch, 10, 512]
        c_s_audio_feat = self.audio_spatial_channel_att(enhanced_visual_feature, enhanced_audio_feature)
        if self.verbose: print('@audio spatial-channel attention: ', c_s_audio_feat.shape) # [batch, 10, 512]
        
        # ============================== Intra-modal Temporal Aware Module ==============================
        # audio_rnn_output, visual_rnn_output = c_s_audio_feat, c_s_visual_feat
        audio_rnn_input, visual_rnn_input = c_s_audio_feat, c_s_visual_feat
        audio_rnn_output, visual_rnn_output = self.audio_visual_rnn_layer(audio_rnn_input, visual_rnn_input)
        audio_encoder_input = audio_rnn_output.transpose(1, 0).contiguous()  # [10, 32, 512]
        visual_encoder_input = visual_rnn_output.transpose(1, 0).contiguous()  # [10, 32, 512]
        if self.verbose: print('@audio encoder input: ', audio_encoder_input.shape) # [10, 32, 512]
        if self.verbose: print('@visual encoder input: ', visual_encoder_input.shape) # [10, 32, 512]

        # ============================== CMRA&ITRM Relation Aware Module ====================================
        # audio query
        visual_key_value_feature = self.visual_encoder(visual_encoder_input)
        if self.verbose: print('@visual key value feature: ', visual_key_value_feature.shape) # [10, 32, 256]
        audio_query_output = self.audio_decoder(audio_encoder_input, visual_key_value_feature)
        if self.verbose: print('@audio query output: ', audio_query_output.shape) # [10, 32, 256]
        
        # visual query
        audio_key_value_feature = self.audio_encoder(audio_encoder_input)
        if self.verbose: print('@audio key value feature: ', audio_key_value_feature.shape) # [10, b, 256]
        visual_query_output = self.visual_decoder(visual_encoder_input, audio_key_value_feature)
        if self.verbose: print('@visual query output: ', visual_query_output.shape) # [10, b, 256]

        # ============================== Cross-Modal Gated Co-Interaction ====================================
        ### ---> unimodal gates
        audio_gate = self.audio_gated(visual_query_output)
        if self.verbose: print('@audio_gate: ', audio_gate.shape)
        visual_gate = self.visual_gated(audio_query_output)
        if self.verbose: print('@visual_gate: ', visual_gate.shape)
        visual_output = (1 - self.alpha) * visual_query_output + audio_gate * visual_query_output * self.alpha
        audio_output = (1 - self.alpha) * audio_query_output + visual_gate * audio_query_output * self.alpha
        # um_gates = torch.sigmoid(audio_gate + visual_gate) 
        um_gates = torch.sigmoid(audio_gate * visual_gate)
        if self.verbose: print('@um_gates: ', um_gates.shape)
        if self.verbose: print('@visual_output: ', visual_output.shape)# torch.Size([10, b, 256])
        if self.verbose: print('@audio_output: ', audio_output.shape)# torch.Size([10, b, 256])

        visual_cas = self.visual_cas(visual_output)  # [10, b, 29]
        visual_cas = visual_cas.permute(1, 0, 2)
        # visual_cas_gate = visual_cas.sigmoid()
        sorted_scores_visual, _ = visual_cas.sort(descending=True, dim=1)
        topk_scores_visual = sorted_scores_visual[:, :4, :]
        score_visual = torch.mean(topk_scores_visual, dim=1)
        if self.verbose: print('@score_visual: ', score_visual.shape)# torch.Size([10, b, 256])
        
        audio_cas = self.audio_cas(audio_output)  # [10, b, 29]
        audio_cas = audio_cas.permute(1, 0, 2)
        # audio_cas_gate = audio_cas.sigmoid()
        sorted_scores_audio, _ = audio_cas.sort(descending=True, dim=1)
        topk_scores_audio = sorted_scores_audio[:, :4, :]
        score_audio = torch.mean(topk_scores_audio, dim=1)  # [b, 28]
        if self.verbose: print('@score_audio: ', score_audio.shape)# torch.Size([10, b, 256])
        
        um_scores =  torch.softmax(score_visual * score_audio, dim=-1) # torch.Size([10, b])
        # um_scores =  torch.sigmoid(score_visual + score_audio) # torch.Size([10, b])
        if self.verbose: print("@um_scores: ", um_scores.shape)
        ### <---

        # ============================== Audio-Visual Interaction Module ====================================
        visual_query_output = self.AVInter(visual_query_output, audio_query_output)
        audio_query_output = self.VAInter(audio_query_output, visual_query_output)
        if self.verbose: print('AVInter@visual_query_output: ', visual_query_output.shape) # [10, b, 256]
        if self.verbose: print('AVInter@audio_query_output: ', audio_query_output.shape) # [10, b, 256]

        ### ---> multimodal gate
        mm_gates = self.global_av_gate(visual_query_output, audio_query_output) # 10x64x1
        if self.verbose: print('@mm_gates: ', mm_gates.shape)
        fused_content = mm_gates * visual_query_output  + (1 - mm_gates) * audio_query_output
        if self.verbose: print('@fused_content: ', fused_content.shape) # ([10, 64, 256])
        mm_gate_cas = self.global_gate_cas(fused_content)
        mm_gate_cas = mm_gate_cas.permute(1, 0, 2)
        sorted_score, _ = mm_gate_cas.sort(descending=True, dim=1)
        topk_gate_scores = sorted_score[:, :4, :]
        mm_scores = torch.mean(topk_gate_scores, dim=1)
        mm_scores = torch.softmax(mm_scores, dim=-1)
        if self.verbose: print('@mm_scores: ', mm_scores.shape) # ([64, 1, 29])
        
        is_event_scores, raw_logits, event_scores = self.localize_module(fused_content)

        ### <---
        return is_event_scores, event_scores, um_gates * mm_gates, um_scores * mm_scores


class supv_main_model(nn.Module):
    def __init__(self, config):
        super(supv_main_model, self).__init__()
        self.config = config
        self.verbose = False
        self.alpha = self.config['alpha']
        self.n_heads = self.config["n_heads"]
        self.visual_input_dim = self.config['visual_inputdim']
        self.audio_input_dim = self.config['audio_inputdim']

        self.visual_fc_dim = 512
        self.d_model = self.config['d_model']

        self.v_fc = nn.Linear(self.visual_input_dim, self.visual_fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        ### ---> multi-head self attention
        self.visual_self_attention = MultiheadAttention(
            embed_dim=self.visual_input_dim, 
            num_heads=self.n_heads, 
            dropout=0.1
            )
        self.audio_self_attention = MultiheadAttention(
            embed_dim=self.audio_input_dim, 
            num_heads=self.n_heads, 
            dropout=0.1
            )
        self.self_att_visual_norm = nn.LayerNorm(self.visual_input_dim)
        self.self_att_audio_norm = nn.LayerNorm(self.audio_input_dim)
        self.self_att_visual_mlp = nn.Sequential(
            nn.Linear(self.visual_input_dim, self.visual_input_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.self_att_audio_mlp = nn.Sequential(
            nn.Linear(self.audio_input_dim, self.audio_input_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        ### <---

        ### ---> spatial-channel attention
        self.visual_spatial_channel_att = A2V_Attention()
        self.audio_spatial_channel_att = V2A_Attention()
        ### <---

        ### ---> temporal relation module
        self.audio_visual_rnn_layer = RNNEncoder(audio_dim=self.visual_input_dim, visual_dim=self.visual_input_dim, d_model=self.d_model, num_layers=1)
        
        self.visual_encoder = InternalTemporalRelationModule(input_dim=self.visual_input_dim, d_model=self.d_model, feedforward_dim=1024)
        self.visual_decoder = CrossModalRelationAttModule(input_dim=self.visual_input_dim, d_model=self.d_model, feedforward_dim=1024)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=512, d_model=self.d_model, feedforward_dim=1024)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=512, d_model=self.d_model, feedforward_dim=1024)

        self.audio_gated = Unimodal_Gate(self.d_model)
        self.visual_gated = Unimodal_Gate(self.d_model)
        self.global_av_gate = Multimodal_Gate(self.d_model * 2)
        self.global_gate_cas = nn.Linear(self.d_model, 28)
        # self.global_gate_cas = CAS_Module(self.d_model, 28)

        self.AVInter = AudiovisualInter(self.d_model, n_head=4, head_dropout=0.2)
        self.VAInter = AudiovisualInter(self.d_model, n_head=4, head_dropout=0.2)
        
        self.visual_norm = nn.LayerNorm(self.d_model)
        self.audio_norm = nn.LayerNorm(self.d_model)

        self.audio_cas = nn.Linear(self.d_model, 28)
        self.visual_cas = nn.Linear(self.d_model, 28)
        self.localize_module = SupvLocalizeModule(self.d_model)

    def forward(self, visual_feature, audio_feature):
        if self.verbose: print('@visual feature: ', visual_feature.shape) # [b, 10, 7, 7, 512]
        if self.verbose: print('@audio feature: ', audio_feature.shape) # [b, 10, 128]

        # ============================== Adaptive Features (Optional) ===============================
        # optional, make the model adaptive to different visual features (e.g., VGG ,ResNet)
        # if feature == 'vgg':
        #     visual_feature = self.v_fc(visual_feature)
        # elif feature == 'resnet':
        # visual_feature = self.v_fc(visual_feature)
        # visual_feature = self.dropout(self.relu(visual_feature))
        # ============================== Self Attention (Visual) ====================================
        b, t, w, h, c = visual_feature.shape
        visual_feature = visual_feature.view(b*t, -1, c) # [b*t, w*h, 512]
        visual_feature = self.self_att_visual_norm(visual_feature)
        visual_attention, _ = self.visual_self_attention(visual_feature, visual_feature, visual_feature) # [b*t, w*h, 512]
        if self.verbose: print('@self attention (visual): ', visual_attention.shape)
        enhanced_visual_feature = self.self_att_visual_norm(visual_feature + visual_attention) # [b*t, h*w, 512]
        enhanced_visual_feature = self.self_att_visual_mlp(enhanced_visual_feature)
        enhanced_visual_feature = enhanced_visual_feature.view(b, t, h, w, c) # [b, t, h, w, 512]
        if self.verbose: print('@enhanced_visual_feature: ', enhanced_visual_feature.shape) # [b, t, h, w, 512]

        # ============================== self Attention (Audio) ======================================
        b, t, c = audio_feature.shape
        audio_feature = audio_feature.view(b, t, -1) # [b, t, 128]
        audio_feature = self.self_att_audio_norm(audio_feature)
        audio_attention, _ = self.audio_self_attention(audio_feature, audio_feature, audio_feature) # [b, t, 128]
        if self.verbose: print('@self attention (audio): ', audio_attention.shape)
        enhanced_audio_feature = self.self_att_audio_norm(audio_feature + audio_attention)
        enhanced_audio_feature = self.self_att_audio_mlp(enhanced_audio_feature)
        if self.verbose: print('@enhanced_audio_feature: ', enhanced_audio_feature.shape) # [b, t, 128]

        # ============================== Spatial Channel Attention ====================================
        c_s_visual_feat = self.visual_spatial_channel_att(enhanced_visual_feature, enhanced_audio_feature)
        if self.verbose: print('@visual spatial-channel attention: ', c_s_visual_feat.shape) # [batch, 10, 512]
        c_s_audio_feat = self.audio_spatial_channel_att(enhanced_visual_feature, enhanced_audio_feature)
        if self.verbose: print('@audio spatial-channel attention: ', c_s_audio_feat.shape) # [batch, 10, 512]
        
        # ============================== Intra-modal Temporal Aware Module ==============================
        # audio_rnn_output, visual_rnn_output = c_s_audio_feat, c_s_visual_feat
        audio_rnn_input, visual_rnn_input = c_s_audio_feat, c_s_visual_feat
        audio_rnn_output, visual_rnn_output = self.audio_visual_rnn_layer(audio_rnn_input, visual_rnn_input)
        audio_encoder_input = audio_rnn_output.transpose(1, 0).contiguous()  # [10, 32, 512]
        visual_encoder_input = visual_rnn_output.transpose(1, 0).contiguous()  # [10, 32, 512]
        if self.verbose: print('@audio encoder input: ', audio_encoder_input.shape) # [10, 32, 512]
        if self.verbose: print('@visual encoder input: ', visual_encoder_input.shape) # [10, 32, 512]

        # ============================== CMRA&ITRM Relation Aware Module ====================================
        # audio query
        visual_key_value_feature = self.visual_encoder(visual_encoder_input)
        if self.verbose: print('@visual key value feature: ', visual_key_value_feature.shape) # [10, 32, 256]
        audio_query_output = self.audio_decoder(audio_encoder_input, visual_key_value_feature)
        if self.verbose: print('@audio query output: ', audio_query_output.shape) # [10, 32, 256]
        
        # visual query
        audio_key_value_feature = self.audio_encoder(audio_encoder_input)
        if self.verbose: print('@audio key value feature: ', audio_key_value_feature.shape) # [10, b, 256]
        visual_query_output = self.visual_decoder(visual_encoder_input, audio_key_value_feature)
        if self.verbose: print('@visual query output: ', visual_query_output.shape) # [10, b, 256]

        # ============================== Audio-Visual Interaction Module ====================================
        visual_query_output = self.AVInter(visual_query_output, audio_query_output).cuda()
        audio_query_output = self.VAInter(audio_query_output, visual_query_output).cuda()
        if self.verbose: print('AVInter@visual_query_output: ', visual_query_output.shape)
        if self.verbose: print('AVInter@audio_query_output: ', audio_query_output.shape)

        # ============================== Modality-aware Calibration Module ====================================
        ### ---> unimodal gate
        audio_gate = self.audio_gated(audio_key_value_feature)
        if self.verbose: print('@audio_gate: ', audio_gate.shape)
        visual_gate = self.visual_gated(visual_key_value_feature)
        if self.verbose: print('@visual_gate: ', visual_gate.shape)
        visual_output = (1 - self.alpha) * visual_query_output + self.alpha * audio_gate * visual_query_output 
        audio_output = (1 - self.alpha) * audio_query_output + self.alpha * visual_gate * audio_query_output   
        um_gates =  audio_gate + visual_gate
        # um_gates =  audio_gate * visual_gate
        # um_gates =  torch.sigmoid(audio_gate + visual_gate) 
        # unimodal_gates = torch.sigmoid(audio_gate * visual_gate)
        if self.verbose: print('@um_gates: ', um_gates.shape)
        if self.verbose: print('@video_output: ', visual_output.shape)# torch.Size([10, b, 256])
        if self.verbose: print('@audio_output: ', audio_output.shape)# torch.Size([10, b, 256])

        visual_cas = self.visual_cas(visual_output)  # [10, b, 28]
        visual_cas = visual_cas.permute(1, 0, 2)
        sorted_scores_visual, _ = visual_cas.sort(descending=True, dim=1)
        topk_scores_visual = sorted_scores_visual[:, :4, :]
        score_visual = torch.mean(topk_scores_visual, dim=1)
        if self.verbose: print('@score_visual: ', score_visual.shape)
        
        audio_cas = self.audio_cas(audio_output)
        audio_cas = audio_cas.permute(1, 0, 2)
        sorted_scores_audio, _ = audio_cas.sort(descending=True, dim=1)
        topk_scores_audio = sorted_scores_audio[:, :4, :]
        score_audio = torch.mean(topk_scores_audio, dim=1)  # [b, 28]
        if self.verbose: print('@score_audio: ', score_audio.shape)

        # um_scores = torch.softmax(score_visual * score_audio)
        # um_scores = score_visual + score_audio
        um_scores = torch.softmax(score_visual + score_audio, dim=-1)
        if self.verbose: print('@um_scores: ', um_scores.shape)
        ### <---

        # multimodal gate
        mm_gates = self.global_av_gate(visual_query_output, audio_query_output) # 10x64x1
        if self.verbose: print('@mm_gates: ', mm_gates.shape)
        visual_output = mm_gates * visual_query_output 
        audio_output = (1 - mm_gates) * audio_query_output
        fused_content = visual_output + audio_output
        if self.verbose: print('@fused_content: ', fused_content.shape) # ([10, 64, 256])
        mm_gate_cas = self.global_gate_cas(fused_content)
        mm_gate_cas = mm_gate_cas.permute(1, 0, 2) 
        sorted_score, _ = mm_gate_cas.sort(descending=True, dim=1)
        topk_gate_scores = sorted_score[:, :4, :]
        mm_scores = torch.mean(topk_gate_scores, dim=1)
        mm_scores = torch.softmax(mm_scores, dim=-1)
        if self.verbose: print('@mm_scores: ', mm_scores.shape) # torch.Size([16, 28])

        is_event_scores, event_scores = self.localize_module(fused_content)
        ### <---

        return is_event_scores, event_scores, um_gates, um_scores, mm_gates, mm_scores
