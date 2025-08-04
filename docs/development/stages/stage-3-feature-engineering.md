# 🔧 Stage 3: Feature Engineering Module

## 📋 CHECKLIST: Feature Engineering (16-20 horas)

### ✅ Prerrequisitos
- [ ] Data Ingestion funcionando y enviando datos
- [ ] Shared libraries operativas
- [ ] Datos de mercado y sentiment fluyendo
- [ ] TimescaleDB con datos históricos

### ✅ Objetivos de la Etapa
Implementar el procesamiento de features para trading:
- **Microstructure Features**: Order book imbalances, delta volume
- **Sentiment Features**: FinBERT processing de tweets
- **Technical Features**: Price momentum, volatility
- **Feature Pipeline**: Normalización y validación
- **Real-time Processing**: Features en tiempo real

## 🏗️ Arquitectura del Módulo

```
modules/feature-engineering/
├── __init__.py
├── main.py                    # Entry point del servicio
├── microstructure/
│   ├── __init__.py
│   ├── order_book_analyzer.py # Análisis order book
│   ├── volume_analyzer.py     # Análisis de volumen
│   └── imbalance_calculator.py # Cálculo de imbalances
├── sentiment/
│   ├── __init__.py
│   ├── finbert_processor.py   # Procesamiento FinBERT
│   ├── sentiment_aggregator.py # Agregación sentiment
│   └── social_metrics.py      # Métricas sociales
├── technical/
│   ├── __init__.py
│   ├── momentum_indicators.py # Indicadores momentum
│   ├── volatility_metrics.py  # Métricas volatilidad
│   └── price_features.py      # Features de precio
├── pipeline/
│   ├── __init__.py
│   ├── feature_processor.py   # Pipeline principal
│   ├── normalizer.py          # Normalización features
│   └── validator.py           # Validación features
└── config/
    └── feature_config.py      # Configuración features
```

## 🚀 Implementación Detallada

### Microstructure Analysis
```python
# modules/feature-engineering/microstructure/order_book_analyzer.py
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from shared.logging.structured_logger import get_logger

logger = get_logger(__name__)

@dataclass
class OrderBookSnapshot:
    timestamp: int
    symbol: str
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]  # [(price, size), ...]

class OrderBookAnalyzer:
    def __init__(self, depth_levels: int = 10):
        self.depth_levels = depth_levels
        self.previous_snapshot = None
        
    def calculate_imbalance(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Calculate order book imbalance metrics"""
        try:
            # Get top levels
            top_bids = snapshot.bids[:self.depth_levels]
            top_asks = snapshot.asks[:self.depth_levels]
            
            if not top_bids or not top_asks:
                return {}
                
            # Calculate bid/ask volumes
            bid_volume = sum(size for _, size in top_bids)
            ask_volume = sum(size for _, size in top_asks)
            
            # Calculate imbalance ratio
            total_volume = bid_volume + ask_volume
            if total_volume == 0:
                imbalance_ratio = 0
            else:
                imbalance_ratio = (bid_volume - ask_volume) / total_volume
                
            # Calculate weighted mid price
            best_bid = top_bids[0][0] if top_bids else 0
            best_ask = top_asks[0][0] if top_asks else 0
            
            if best_bid > 0 and best_ask > 0:
                spread = best_ask - best_bid
                spread_pct = spread / ((best_bid + best_ask) / 2) * 100
                weighted_mid = (best_bid * ask_volume + best_ask * bid_volume) / total_volume
            else:
                spread = 0
                spread_pct = 0
                weighted_mid = 0
                
            # Calculate depth imbalance at different levels
            level_imbalances = {}
            for level in [1, 3, 5, 10]:
                if level <= len(top_bids) and level <= len(top_asks):
                    bid_vol_level = sum(size for _, size in top_bids[:level])
                    ask_vol_level = sum(size for _, size in top_asks[:level])
                    total_vol_level = bid_vol_level + ask_vol_level
                    
                    if total_vol_level > 0:
                        level_imbalances[f'imbalance_L{level}'] = (bid_vol_level - ask_vol_level) / total_vol_level
                    else:
                        level_imbalances[f'imbalance_L{level}'] = 0
                        
            return {
                'timestamp': snapshot.timestamp,
                'symbol': snapshot.symbol,
                'imbalance_ratio': imbalance_ratio,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'spread': spread,
                'spread_pct': spread_pct,
                'weighted_mid': weighted_mid,
                **level_imbalances
            }
            
        except Exception as e:
            logger.error(f"Error calculating order book imbalance: {e}")
            return {}
            
    def calculate_flow_toxicity(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Calculate order flow toxicity (VPIN-like metric)"""
        if self.previous_snapshot is None:
            self.previous_snapshot = snapshot
            return {}
            
        try:
            # Calculate volume imbalance between snapshots
            current_mid = self._get_mid_price(snapshot)
            previous_mid = self._get_mid_price(self.previous_snapshot)
            
            if current_mid == 0 or previous_mid == 0:
                return {}
                
            price_change = (current_mid - previous_mid) / previous_mid
            
            # Estimate buy/sell volumes based on price movement
            total_volume = sum(size for _, size in snapshot.bids + snapshot.asks)
            
            if price_change > 0:
                # Price went up, more buying pressure
                buy_volume = total_volume * (0.5 + abs(price_change))
                sell_volume = total_volume - buy_volume
            else:
                # Price went down, more selling pressure
                sell_volume = total_volume * (0.5 + abs(price_change))
                buy_volume = total_volume - sell_volume
                
            # Calculate VPIN (Volume-Synchronized Probability of Informed Trading)
            if total_volume > 0:
                vpin = abs(buy_volume - sell_volume) / total_volume
            else:
                vpin = 0
                
            self.previous_snapshot = snapshot
            
            return {
                'timestamp': snapshot.timestamp,
                'symbol': snapshot.symbol,
                'vpin': vpin,
                'buy_volume_est': buy_volume,
                'sell_volume_est': sell_volume,
                'price_change': price_change
            }
            
        except Exception as e:
            logger.error(f"Error calculating flow toxicity: {e}")
            return {}
            
    def _get_mid_price(self, snapshot: OrderBookSnapshot) -> float:
        """Get mid price from order book"""
        if not snapshot.bids or not snapshot.asks:
            return 0
            
        best_bid = snapshot.bids[0][0]
        best_ask = snapshot.asks[0][0]
        
        return (best_bid + best_ask) / 2
```

### Sentiment Analysis with FinBERT
```python
# modules/feature-engineering/sentiment/finbert_processor.py
import asyncio
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from shared.logging.structured_logger import get_logger

logger = get_logger(__name__)

class FinBERTProcessor:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    async def initialize(self):
        """Initialize FinBERT model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"FinBERT model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize FinBERT: {e}")
            raise
            
    async def process_text(self, text: str) -> Dict[str, float]:
        """Process single text for sentiment"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("FinBERT model not initialized")
            
        try:
            # Clean and prepare text
            cleaned_text = self._clean_text(text)
            
            # Tokenize
            inputs = self.tokenizer(
                cleaned_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Convert to probabilities
            probs = predictions.cpu().numpy()[0]
            
            # FinBERT classes: negative, neutral, positive
            sentiment_scores = {
                'negative': float(probs[0]),
                'neutral': float(probs[1]),
                'positive': float(probs[2])
            }
            
            # Calculate compound score
            compound_score = sentiment_scores['positive'] - sentiment_scores['negative']
            
            return {
                'sentiment_scores': sentiment_scores,
                'compound_score': compound_score,
                'confidence': float(np.max(probs)),
                'text_length': len(cleaned_text)
            }
            
        except Exception as e:
            logger.error(f"Error processing text sentiment: {e}")
            return {
                'sentiment_scores': {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33},
                'compound_score': 0.0,
                'confidence': 0.0,
                'text_length': 0
            }
            
    async def process_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Process batch of texts"""
        results = []
        
        # Process in smaller batches to avoid memory issues
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self.process_text(text) for text in batch]
            )
            results.extend(batch_results)
            
        return results
        
    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        import re
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (but keep the text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove non-ASCII characters
        text = ''.join(char for char in text if ord(char) < 128)
        
        return text

class SentimentAggregator:
    def __init__(self, window_minutes: int = 5):
        self.window_minutes = window_minutes
        self.sentiment_buffer = []
        
    def add_sentiment(self, sentiment_data: Dict[str, Any]):
        """Add sentiment data to buffer"""
        self.sentiment_buffer.append(sentiment_data)
        
        # Keep only recent data
        current_time = sentiment_data.get('timestamp', 0)
        cutoff_time = current_time - (self.window_minutes * 60 * 1000)  # milliseconds
        
        self.sentiment_buffer = [
            s for s in self.sentiment_buffer 
            if s.get('timestamp', 0) > cutoff_time
        ]
        
    def get_aggregated_sentiment(self) -> Dict[str, float]:
        """Get aggregated sentiment metrics"""
        if not self.sentiment_buffer:
            return {
                'avg_compound': 0.0,
                'avg_positive': 0.33,
                'avg_negative': 0.33,
                'avg_neutral': 0.34,
                'sentiment_momentum': 0.0,
                'tweet_count': 0,
                'avg_confidence': 0.0
            }
            
        # Calculate averages
        compound_scores = [s.get('compound_score', 0) for s in self.sentiment_buffer]
        positive_scores = [s.get('sentiment_scores', {}).get('positive', 0) for s in self.sentiment_buffer]
        negative_scores = [s.get('sentiment_scores', {}).get('negative', 0) for s in self.sentiment_buffer]
        neutral_scores = [s.get('sentiment_scores', {}).get('neutral', 0) for s in self.sentiment_buffer]
        confidence_scores = [s.get('confidence', 0) for s in self.sentiment_buffer]
        
        # Calculate momentum (recent vs older sentiment)
        if len(compound_scores) >= 2:
            recent_half = compound_scores[len(compound_scores)//2:]
            older_half = compound_scores[:len(compound_scores)//2]
            
            recent_avg = np.mean(recent_half) if recent_half else 0
            older_avg = np.mean(older_half) if older_half else 0
            
            sentiment_momentum = recent_avg - older_avg
        else:
            sentiment_momentum = 0.0
            
        return {
            'avg_compound': np.mean(compound_scores),
            'avg_positive': np.mean(positive_scores),
            'avg_negative': np.mean(negative_scores),
            'avg_neutral': np.mean(neutral_scores),
            'sentiment_momentum': sentiment_momentum,
            'tweet_count': len(self.sentiment_buffer),
            'avg_confidence': np.mean(confidence_scores),
            'sentiment_volatility': np.std(compound_scores) if len(compound_scores) > 1 else 0.0
        }
```

### Feature Pipeline
```python
# modules/feature-engineering/pipeline/feature_processor.py
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from shared.messaging.message_broker import HybridMessageBroker
from shared.database.connection import DatabaseManager
from shared.logging.structured_logger import get_logger
from microstructure.order_book_analyzer import OrderBookAnalyzer, OrderBookSnapshot
from sentiment.finbert_processor import FinBERTProcessor, SentimentAggregator
from technical.momentum_indicators import MomentumCalculator
from pipeline.normalizer import FeatureNormalizer

logger = get_logger(__name__)

@dataclass
class FeatureVector:
    timestamp: int
    symbol: str
    
    # Microstructure features
    imbalance_ratio: float = 0.0
    imbalance_L1: float = 0.0
    imbalance_L3: float = 0.0
    imbalance_L5: float = 0.0
    imbalance_L10: float = 0.0
    spread_pct: float = 0.0
    vpin: float = 0.0
    
    # Sentiment features
    sentiment_compound: float = 0.0
    sentiment_positive: float = 0.0
    sentiment_negative: float = 0.0
    sentiment_momentum: float = 0.0
    sentiment_volatility: float = 0.0
    tweet_count: float = 0.0
    
    # Technical features
    price_momentum_1m: float = 0.0
    price_momentum_5m: float = 0.0
    price_momentum_15m: float = 0.0
    volatility_1m: float = 0.0
    volatility_5m: float = 0.0
    rsi_14: float = 50.0
    
    # Meta features
    market_hour: int = 0
    day_of_week: int = 0
    volume_profile: float = 0.0

class FeatureProcessor:
    def __init__(self, message_broker: HybridMessageBroker, db_manager: DatabaseManager):
        self.message_broker = message_broker
        self.db_manager = db_manager
        
        # Initialize processors
        self.order_book_analyzer = OrderBookAnalyzer()
        self.finbert_processor = FinBERTProcessor()
        self.sentiment_aggregator = SentimentAggregator()
        self.momentum_calculator = MomentumCalculator()
        self.feature_normalizer = FeatureNormalizer()
        
        # Feature storage
        self.latest_features = {}
        self.feature_history = []
        
    async def start(self):
        """Start feature processing service"""
        try:
            # Initialize FinBERT
            await self.finbert_processor.initialize()
            
            # Subscribe to data streams
            await self.message_broker.subscribe_fast(
                ['order_book', 'market_data', 'social_data'],
                'feature-engineering',
                'processor-1',
                self._process_message
            )
            
            # Start feature generation loop
            asyncio.create_task(self._feature_generation_loop())
            
            logger.info("Feature processor started")
            
        except Exception as e:
            logger.error(f"Failed to start feature processor: {e}")
            raise
            
    async def _process_message(self, stream: str, message_id: str, data: Dict[str, Any]):
        """Process incoming data messages"""
        try:
            if stream == 'order_book':
                await self._process_order_book(data)
            elif stream == 'market_data':
                await self._process_market_data(data)
            elif stream == 'social_data':
                await self._process_social_data(data)
                
        except Exception as e:
            logger.error(f"Error processing message from {stream}: {e}")
            
    async def _process_order_book(self, data: Dict[str, Any]):
        """Process order book data"""
        try:
            # Convert to OrderBookSnapshot
            snapshot = OrderBookSnapshot(
                timestamp=data['timestamp'],
                symbol=data['symbol'],
                bids=data['bids'],
                asks=data['asks']
            )
            
            # Calculate microstructure features
            imbalance_features = self.order_book_analyzer.calculate_imbalance(snapshot)
            toxicity_features = self.order_book_analyzer.calculate_flow_toxicity(snapshot)
            
            # Store features
            self.latest_features.update({
                'microstructure': {**imbalance_features, **toxicity_features}
            })
            
        except Exception as e:
            logger.error(f"Error processing order book: {e}")
            
    async def _process_market_data(self, data: Dict[str, Any]):
        """Process market data"""
        try:
            # Calculate technical indicators
            technical_features = await self.momentum_calculator.calculate_features(data)
            
            # Store features
            self.latest_features.update({
                'technical': technical_features
            })
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            
    async def _process_social_data(self, data: Dict[str, Any]):
        """Process social media data"""
        try:
            # Process sentiment
            sentiment_result = await self.finbert_processor.process_text(data['text'])
            
            # Add to aggregator
            sentiment_data = {
                'timestamp': data['timestamp'],
                **sentiment_result
            }
            self.sentiment_aggregator.add_sentiment(sentiment_data)
            
            # Get aggregated sentiment
            aggregated_sentiment = self.sentiment_aggregator.get_aggregated_sentiment()
            
            # Store features
            self.latest_features.update({
                'sentiment': aggregated_sentiment
            })
            
        except Exception as e:
            logger.error(f"Error processing social data: {e}")
            
    async def _feature_generation_loop(self):
        """Main feature generation loop"""
        while True:
            try:
                await asyncio.sleep(1)  # Generate features every second
                
                if self._has_sufficient_data():
                    feature_vector = await self._generate_feature_vector()
                    
                    if feature_vector:
                        # Normalize features
                        normalized_features = await self.feature_normalizer.normalize(feature_vector)
                        
                        # Publish features
                        await self.message_broker.publish_fast(
                            'features',
                            asdict(normalized_features)
                        )
                        
                        # Store in database
                        await self._store_features(normalized_features)
                        
            except Exception as e:
                logger.error(f"Error in feature generation loop: {e}")
                
    def _has_sufficient_data(self) -> bool:
        """Check if we have sufficient data to generate features"""
        required_keys = ['microstructure', 'technical', 'sentiment']
        return all(key in self.latest_features for key in required_keys)
        
    async def _generate_feature_vector(self) -> Optional[FeatureVector]:
        """Generate complete feature vector"""
        try:
            import time
            from datetime import datetime
            
            current_time = int(time.time() * 1000)
            dt = datetime.fromtimestamp(current_time / 1000)
            
            # Extract features from latest data
            micro_features = self.latest_features.get('microstructure', {})
            tech_features = self.latest_features.get('technical', {})
            sent_features = self.latest_features.get('sentiment', {})
            
            feature_vector = FeatureVector(
                timestamp=current_time,
                symbol='BTCUSDT',
                
                # Microstructure
                imbalance_ratio=micro_features.get('imbalance_ratio', 0.0),
                imbalance_L1=micro_features.get('imbalance_L1', 0.0),
                imbalance_L3=micro_features.get('imbalance_L3', 0.0),
                imbalance_L5=micro_features.get('imbalance_L5', 0.0),
                imbalance_L10=micro_features.get('imbalance_L10', 0.0),
                spread_pct=micro_features.get('spread_pct', 0.0),
                vpin=micro_features.get('vpin', 0.0),
                
                # Sentiment
                sentiment_compound=sent_features.get('avg_compound', 0.0),
                sentiment_positive=sent_features.get('avg_positive', 0.33),
                sentiment_negative=sent_features.get('avg_negative', 0.33),
                sentiment_momentum=sent_features.get('sentiment_momentum', 0.0),
                sentiment_volatility=sent_features.get('sentiment_volatility', 0.0),
                tweet_count=sent_features.get('tweet_count', 0.0),
                
                # Technical
                price_momentum_1m=tech_features.get('momentum_1m', 0.0),
                price_momentum_5m=tech_features.get('momentum_5m', 0.0),
                price_momentum_15m=tech_features.get('momentum_15m', 0.0),
                volatility_1m=tech_features.get('volatility_1m', 0.0),
                volatility_5m=tech_features.get('volatility_5m', 0.0),
                rsi_14=tech_features.get('rsi_14', 50.0),
                
                # Meta
                market_hour=dt.hour,
                day_of_week=dt.weekday(),
                volume_profile=tech_features.get('volume_profile', 0.0)
            )
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error generating feature vector: {e}")
            return None
            
    async def _store_features(self, features: FeatureVector):
        """Store features in database"""
        try:
            query = """
                INSERT INTO trading.features (
                    timestamp, symbol, imbalance_ratio, imbalance_l1, imbalance_l3,
                    imbalance_l5, imbalance_l10, spread_pct, vpin, sentiment_compound,
                    sentiment_positive, sentiment_negative, sentiment_momentum,
                    sentiment_volatility, tweet_count, price_momentum_1m,
                    price_momentum_5m, price_momentum_15m, volatility_1m,
                    volatility_5m, rsi_14, market_hour, day_of_week, volume_profile
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                    $15, $16, $17, $18, $19, $20, $21, $22, $23, $24
                )
            """
            
            await self.db_manager.execute(
                query,
                features.timestamp, features.symbol, features.imbalance_ratio,
                features.imbalance_L1, features.imbalance_L3, features.imbalance_L5,
                features.imbalance_L10, features.spread_pct, features.vpin,
                features.sentiment_compound, features.sentiment_positive,
                features.sentiment_negative, features.sentiment_momentum,
                features.sentiment_volatility, features.tweet_count,
                features.price_momentum_1m, features.price_momentum_5m,
                features.price_momentum_15m, features.volatility_1m,
                features.volatility_5m, features.rsi_14, features.market_hour,
                features.day_of_week, features.volume_profile
            )
            
        except Exception as e:
            logger.error(f"Error storing features: {e}")
```

## ✅ Testing y Validación

### Feature Quality Tests
```python
# tests/unit/test_feature_quality.py
import pytest
import numpy as np
from modules.feature_engineering.pipeline.feature_processor import FeatureVector

def test_feature_vector_completeness():
    """Test that feature vector has all required fields"""
    feature_vector = FeatureVector(
        timestamp=1234567890,
        symbol='BTCUSDT'
    )
    
    # Check all fields exist
    required_fields = [
        'imbalance_ratio', 'sentiment_compound', 'price_momentum_1m',
        'volatility_1m', 'market_hour', 'day_of_week'
    ]
    
    for field in required_fields:
        assert hasattr(feature_vector, field)
        
def test_feature_normalization():
    """Test feature normalization"""
    from modules.feature_engineering.pipeline.normalizer import FeatureNormalizer
    
    normalizer = FeatureNormalizer()
    
    # Test data
    features = FeatureVector(
        timestamp=1234567890,
        symbol='BTCUSDT',
        imbalance_ratio=0.5,
        sentiment_compound=0.8,
        price_momentum_1m=0.02
    )
    
    normalized = normalizer.normalize(features)
    
    # Check normalization bounds
    assert -3 <= normalized.imbalance_ratio <= 3
    assert -3 <= normalized.sentiment_compound <= 3
```

## ✅ Checklist de Completitud

### Microstructure Features
- [ ] Order book imbalance calculation
- [ ] Multi-level depth analysis
- [ ] VPIN toxicity metric
- [ ] Spread analysis
- [ ] Volume flow analysis
- [ ] Unit tests pasando

### Sentiment Features
- [ ] FinBERT integration funcionando
- [ ] Text preprocessing pipeline
- [ ] Sentiment aggregation
- [ ] Momentum calculation
- [ ] Confidence scoring
- [ ] Batch processing optimizado

### Technical Features
- [ ] Price momentum indicators
- [ ] Volatility metrics
- [ ] RSI calculation
- [ ] Volume profile analysis
- [ ] Time-based features
- [ ] Historical data integration

### Feature Pipeline
- [ ] Real-time processing
- [ ] Feature normalization
- [ ] Data validation
- [ ] Storage optimization
- [ ] Error handling robusto
- [ ] Performance monitoring

**Tiempo estimado**: 16-20 horas  
**Responsable**: ML Engineer + Data Scientist  
**Dependencias**: Data Ingestion funcionando

---

**Next Step**: [Stage 4: Signal Generation](./stage-4-signal-generation.md)
