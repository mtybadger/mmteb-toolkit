from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
import os
from datasets import load_dataset

class RedditClusteringDE(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "RedditClustering",
            "hf_hub_name": "willhath/german-reddit-clustering",
            "description": (
                "Clustering of titles from 199 subreddits. Clustering of 25 sets, each with 10-50 classes, and each"
                " class with 100 - 1000 sentences."
            ),
            "reference": "https://arxiv.org/abs/2104.07081",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["de"],
            "main_score": "v_measure",
            "revision": "68b429afe701f1ff0fa65fcbb1b4a186ba89da43",
        }
    
class RedditClusteringES(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "RedditClustering",
            "hf_hub_name": "willhath/spanish-reddit-clustering",
            "description": (
                "Clustering of titles from 199 subreddits. Clustering of 25 sets, each with 10-50 classes, and each"
                " class with 100 - 1000 sentences."
            ),
            "reference": "https://arxiv.org/abs/2104.07081",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["es"],
            "main_score": "v_measure",
            "revision": "1e419fcdc7a45c5b8953e6429746ffe64e41a831",
        }
    
class RedditClusteringFR(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "RedditClustering",
            "hf_hub_name": "willhath/french-reddit-clustering",
            "description": (
                "Clustering of titles from 199 subreddits. Clustering of 25 sets, each with 10-50 classes, and each"
                " class with 100 - 1000 sentences."
            ),
            "reference": "https://arxiv.org/abs/2104.07081",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["fr"],
            "main_score": "v_measure",
            "revision": "600231288f9b4b1a59f63ffddc9f5477277018cf",
        }
    
class RedditClusteringTR(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "RedditClustering",
            "hf_hub_name": "willhath/turkish-reddit-clustering",
            "description": (
                "Clustering of titles from 199 subreddits. Clustering of 25 sets, each with 10-50 classes, and each"
                " class with 100 - 1000 sentences."
            ),
            "reference": "https://arxiv.org/abs/2104.07081",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["tr"],
            "main_score": "v_measure",
            "revision": "216945c31bcb1e3d32f26cf77e0ee1eb1376a0d9",
        }
    
class RedditClusteringSW(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "RedditClustering",
            "hf_hub_name": "willhath/swahili-reddit-clustering",
            "description": (
                "Clustering of titles from 199 subreddits. Clustering of 25 sets, each with 10-50 classes, and each"
                " class with 100 - 1000 sentences."
            ),
            "reference": "https://arxiv.org/abs/2104.07081",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["sw"],
            "main_score": "v_measure",
            "revision": "d68683f8c508601b492022e67789a8ab56a430f2",
        }
    
class TwentyNewsgroupsClusteringDE(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "TwentyNewsgroupsClustering",
            "hf_hub_name": "willhath/german-twentynewsgroups-clustering",
            "description": "Clustering of the 20 Newsgroups dataset (subject only).",
            "reference": "https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["de"],
            "main_score": "v_measure",
            "revision": "b4c12bc12dc3f917f846703a9ff7dd32beb518aa",
        }
    
class TwentyNewsgroupsClusteringES(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "TwentyNewsgroupsClustering",
            "hf_hub_name": "willhath/spanish-twentynewsgroups-clustering",
            "description": "Clustering of the 20 Newsgroups dataset (subject only).",
            "reference": "https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["es"],
            "main_score": "v_measure",
            "revision": "f3b553cd2a54d404bea02985f4065834211f3abe",
        }
    
class TwentyNewsgroupsClusteringFR(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "TwentyNewsgroupsClustering",
            "hf_hub_name": "willhath/french-twentynewsgroups-clustering",
            "description": "Clustering of the 20 Newsgroups dataset (subject only).",
            "reference": "https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["fr"],
            "main_score": "v_measure",
            "revision": "b32987cd0f9a7f000aa6293e48efea38e9038794",
        }
    
class TwentyNewsgroupsClusteringTR(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "TwentyNewsgroupsClustering",
            "hf_hub_name": "willhath/turkish-twentynewsgroups-clustering",
            "description": "Clustering of the 20 Newsgroups dataset (subject only).",
            "reference": "https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["tr"],
            "main_score": "v_measure",
            "revision": "448f38e48880a6386dccf647a2eebc078a1c1e2e",
        }
    
class TwentyNewsgroupsClusteringSW(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "TwentyNewsgroupsClustering",
            "hf_hub_name": "willhath/swahili-twentynewsgroups-clustering",
            "description": "Clustering of the 20 Newsgroups dataset (subject only).",
            "reference": "https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["sw"],
            "main_score": "v_measure",
            "revision": "d349bb48e733f237fe47a8923f3dd718000e0985",
        }
    

from mteb.abstasks.AbsTaskSummarization import AbsTaskSummarization

class SummEvalFR(AbsTaskSummarization):
    @property
    def description(self):
        return {
            "name": "SummEval",
            "hf_hub_name": "sproos/summeval-fr",
            "description": "News Article Summary Semantic Similarity Estimation.",
            "reference": "https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html",
            "type": "Summarization",
            "category": "p2p",
            "eval_splits": ["test"],
            "eval_langs": ["fr"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
            "revision": "6c9e6944ea8ec20291a1488eb762f32a438cd6ae",
        }
    
class SummEvalDE(AbsTaskSummarization):
    @property
    def description(self):
        return {
            "name": "SummEval",
            "hf_hub_name": "sproos/summeval-de",
            "description": "News Article Summary Semantic Similarity Estimation.",
            "reference": "https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html",
            "type": "Summarization",
            "category": "p2p",
            "eval_splits": ["train"],
            "eval_langs": ["de"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
            "revision": "d25b21e407a2d22c96681fe3ae700eb2cc0c9bb9",
        }
    
class SummEvalES(AbsTaskSummarization):
    @property
    def description(self):
        return {
            "name": "SummEval",
            "hf_hub_name": "sproos/summeval-es",
            "description": "News Article Summary Semantic Similarity Estimation.",
            "reference": "https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html",
            "type": "Summarization",
            "category": "p2p",
            "eval_splits": ["train"],
            "eval_langs": ["es"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
            "revision": "806c651ffcc893ee518a2b198fa160d973e2182b",
        }
    
class SummEvalTR(AbsTaskSummarization):
    @property
    def description(self):
        return {
            "name": "SummEval",
            "hf_hub_name": "sproos/summeval-tr",
            "description": "News Article Summary Semantic Similarity Estimation.",
            "reference": "https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html",
            "type": "Summarization",
            "category": "p2p",
            "eval_splits": ["train"],
            "eval_langs": ["tr"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
            "revision": "d773054c462a22de9a3d4088ab49890bba06118f",
        }
    
class SummEvalSW(AbsTaskSummarization):
    @property
    def description(self):
        return {
            "name": "SummEval",
            "hf_hub_name": "sproos/summeval-sw",
            "description": "News Article Summary Semantic Similarity Estimation.",
            "reference": "https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html",
            "type": "Summarization",
            "category": "p2p",
            "eval_splits": ["train"],
            "eval_langs": ["sw"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
            "revision": "4eb1c5a78fa8f0e9236c1c969d399c3834944e1e",
        }
    
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval

class SciFactES(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "SciFact",
            "description": "SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.",
            "reference": "https://github.com/allenai/scifact",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["es"],
            "main_score": "ndcg_at_10",
        }
    
    def load_data(self, eval_splits=None, **kwargs):
        """
        Load dataset from BeIR benchmark. TODO: replace with HF hub once datasets are moved there
        """
        from beir.datasets.data_loader import GenericDataLoader 
        if self.data_loaded:
            return
        if eval_splits is None:
            eval_splits = self.description["eval_splits"]
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        for split in eval_splits:
            self.corpus[split], self.queries[split], self.relevant_docs[split] = GenericDataLoader(
                data_folder='./scifact_es/'
            ).load(split=split)
        self.data_loaded = True

class SciFactFR(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "SciFact",
            "description": "SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.",
            "reference": "https://github.com/allenai/scifact",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["fr"],
            "main_score": "ndcg_at_10",
        }
    
    def load_data(self, eval_splits=None, **kwargs):
        """
        Load dataset from BeIR benchmark. TODO: replace with HF hub once datasets are moved there
        """
        from beir.datasets.data_loader import GenericDataLoader 
        if self.data_loaded:
            return
        if eval_splits is None:
            eval_splits = self.description["eval_splits"]
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        for split in eval_splits:
            self.corpus[split], self.queries[split], self.relevant_docs[split] = GenericDataLoader(
                data_folder='./scifact_fr/'
            ).load(split=split)
        self.data_loaded = True

class SciFactTR(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "SciFact",
            "description": "SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.",
            "reference": "https://github.com/allenai/scifact",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["tr"],
            "main_score": "ndcg_at_10",
        }
    
    def load_data(self, eval_splits=None, **kwargs):
        """
        Load dataset from BeIR benchmark. TODO: replace with HF hub once datasets are moved there
        """
        from beir.datasets.data_loader import GenericDataLoader 
        if self.data_loaded:
            return
        if eval_splits is None:
            eval_splits = self.description["eval_splits"]
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        for split in eval_splits:
            self.corpus[split], self.queries[split], self.relevant_docs[split] = GenericDataLoader(
                data_folder='./scifact_tr/'
            ).load(split=split)
        self.data_loaded = True

class SciFactDE(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "SciFact",
            "description": "SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.",
            "reference": "https://github.com/allenai/scifact",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["de"],
            "main_score": "ndcg_at_10",
        }
    
    def load_data(self, eval_splits=None, **kwargs):
        """
        Load dataset from BeIR benchmark. TODO: replace with HF hub once datasets are moved there
        """
        from beir.datasets.data_loader import GenericDataLoader 
        if self.data_loaded:
            return
        if eval_splits is None:
            eval_splits = self.description["eval_splits"]
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        for split in eval_splits:
            self.corpus[split], self.queries[split], self.relevant_docs[split] = GenericDataLoader(
                data_folder='./scifact_de/'
            ).load(split=split)
        self.data_loaded = True

class SciFactSW(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "SciFact",
            "description": "SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.",
            "reference": "https://github.com/allenai/scifact",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["sw"],
            "main_score": "ndcg_at_10",
        }
    
    def load_data(self, eval_splits=None, **kwargs):
        """
        Load dataset from BeIR benchmark. TODO: replace with HF hub once datasets are moved there
        """
        from beir.datasets.data_loader import GenericDataLoader 
        if self.data_loaded:
            return
        if eval_splits is None:
            eval_splits = self.description["eval_splits"]
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        for split in eval_splits:
            self.corpus[split], self.queries[split], self.relevant_docs[split] = GenericDataLoader(
                data_folder='./scifact_sw/'
            ).load(split=split)
        self.data_loaded = True
        
from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification

class TwitterURLCorpusPCES(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "TwitterURLCorpus",
            "hf_hub_name": "sproos/twitter-pairclass-es",
            "description": "Paraphrase-Pairs of Tweets.",
            "reference": "https://languagenet.github.io/",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["train"],
            "eval_langs": ["es"],
            "main_score": "ap",
            "revision": "270dc1d8c9a3e515bf727e69efb30ce6903d0d87",
        }
    
class TwitterURLCorpusPCFR(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "TwitterURLCorpus",
            "hf_hub_name": "sproos/twitter-pairclass-fr",
            "description": "Paraphrase-Pairs of Tweets.",
            "reference": "https://languagenet.github.io/",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["train"],
            "eval_langs": ["fr"],
            "main_score": "ap",
            "revision": "bbd7a66e7fcf140d5a0d92d3234f609a6d06298d",
        }

class TwitterURLCorpusPCDE(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "TwitterURLCorpus",
            "hf_hub_name": "sproos/twitter-pairclass-de",
            "description": "Paraphrase-Pairs of Tweets.",
            "reference": "https://languagenet.github.io/",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["train"],
            "eval_langs": ["de"],
            "main_score": "ap",
            "revision": "310df3c8ea10c127d5735e63245dc0b85746bc28",
        }
    
class TwitterURLCorpusPCTR(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "TwitterURLCorpus",
            "hf_hub_name": "sproos/twitter-pairclass-tr",
            "description": "Paraphrase-Pairs of Tweets.",
            "reference": "https://languagenet.github.io/",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["train"],
            "eval_langs": ["tr"],
            "main_score": "ap",
            "revision": "feb67a7bf8141236513594e1ee122325f9d45026",
        }
    
class TwitterURLCorpusPCSW(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            "name": "TwitterURLCorpus",
            "hf_hub_name": "sproos/twitter-pairclass-sw",
            "description": "Paraphrase-Pairs of Tweets.",
            "reference": "https://languagenet.github.io/",
            "category": "s2s",
            "type": "PairClassification",
            "eval_splits": ["train"],
            "eval_langs": ["sw"],
            "main_score": "ap",
            "revision": "c0ff83c2b9b023ecf3ffe9d68f258a8198b9a718",
        }

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking

class MindSmallRerankingES(AbsTaskReranking):
    @property
    def description(self):
        return {
            "name": "MindSmallReranking",
            "hf_hub_name": "sproos/mindsmall-es",
            "description": "Microsoft News Dataset: A Large-Scale English Dataset for News Recommendation Research",
            "reference": "https://msnews.github.io/assets/doc/ACL2020_MIND.pdf",
            "type": "Reranking",
            "category": "s2s",
            "eval_splits": ["train"],
            "eval_langs": ["es"],
            "main_score": "map",
            "revision": "4b8496471b305dd89be11b3785b9d37649bc302e",
        }
    
class MindSmallRerankingDE(AbsTaskReranking):
    @property
    def description(self):
        return {
            "name": "MindSmallReranking",
            "hf_hub_name": "sproos/mindsmall-de",
            "description": "Microsoft News Dataset: A Large-Scale English Dataset for News Recommendation Research",
            "reference": "https://msnews.github.io/assets/doc/ACL2020_MIND.pdf",
            "type": "Reranking",
            "category": "s2s",
            "eval_splits": ["train"],
            "eval_langs": ["de"],
            "main_score": "map",
            "revision": "7850f8aa5a1220d2d844aa8c9076d2e259278657",
        }

class MindSmallRerankingFR(AbsTaskReranking):
    @property
    def description(self):
        return {
            "name": "MindSmallReranking",
            "hf_hub_name": "sproos/mindsmall-fr",
            "description": "Microsoft News Dataset: A Large-Scale English Dataset for News Recommendation Research",
            "reference": "https://msnews.github.io/assets/doc/ACL2020_MIND.pdf",
            "type": "Reranking",
            "category": "s2s",
            "eval_splits": ["train"],
            "eval_langs": ["fr"],
            "main_score": "map",
            "revision": "7850f8aa5a1220d2d844aa8c9076d2e259278657",
        }

class MindSmallRerankingTR(AbsTaskReranking):
    @property
    def description(self):
        return {
            "name": "MindSmallReranking",
            "hf_hub_name": "sproos/mindsmall-tr",
            "description": "Microsoft News Dataset: A Large-Scale English Dataset for News Recommendation Research",
            "reference": "https://msnews.github.io/assets/doc/ACL2020_MIND.pdf",
            "type": "Reranking",
            "category": "s2s",
            "eval_splits": ["train"],
            "eval_langs": ["tr"],
            "main_score": "map",
            "revision": "ecfd78dd48627f75e411cf93177b1f95aa90bd11",
        }
    
class MindSmallRerankingSW(AbsTaskReranking):
    @property
    def description(self):
        return {
            "name": "MindSmallReranking",
            "hf_hub_name": "sproos/mindsmall-sw",
            "description": "Microsoft News Dataset: A Large-Scale English Dataset for News Recommendation Research",
            "reference": "https://msnews.github.io/assets/doc/ACL2020_MIND.pdf",
            "type": "Reranking",
            "category": "s2s",
            "eval_splits": ["train"],
            "eval_langs": ["sw"],
            "main_score": "map",
            "revision": "ecfd78dd48627f75e411cf93177b1f95aa90bd11",
        }