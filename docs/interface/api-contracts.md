# 🔌 API Contracts - Backend to Frontend

## 📋 Overview

Contratos de API detallados que cada módulo backend debe exponer para la interfaz Next.js. Incluye endpoints, WebSocket events, data schemas y error handling.

---

## 🏗️ ETAPA 0: Interface Base

### **System API**
```typescript
GET /api/system/status
Response: {
  status: 'setup' | 'developing' | 'active' | 'error'
  version: string
  uptime: number
  environment: 'development' | 'production'
}

GET /api/system/config
Response: {
  configPath: string
  hardwareProfile: HardwareProfile
  modules: ModuleStatus[]
}

POST /api/system/config
Body: { configPath: string, config: Record<string, any> }
Response: { success: boolean, errors?: string[] }
```

---

## 🔧 ETAPA 1: Shared Libraries + System Monitoring

### **System Health API**
```typescript
GET /api/system/resources
Response: {
  cpu: { usage: number, cores: number, temperature?: number }
  memory: { used: number, total: number, percentage: number }
  disk: { used: number, total: number, percentage: number }
  timestamp: string
}

GET /api/system/services
Response: {
  services: ServiceInfo[]
}

interface ServiceInfo {
  id: string
  name: string
  status: 'running' | 'stopped' | 'error' | 'starting'
  uptime: number
  memoryUsage: number
  cpuUsage: number
  healthCheck: {
    status: 'healthy' | 'unhealthy' | 'unknown'
    lastCheck: string
    responseTime?: number
  }
}

POST /api/system/services/{serviceId}/restart
Response: { success: boolean, message: string }
```

### **Configuration API**
```typescript
GET /api/config
Response: {
  config: Record<string, any>
  schema?: JSONSchema
  lastModified: string
}

PUT /api/config
Body: { config: Record<string, any> }
Response: { 
  success: boolean
  errors?: ValidationError[]
  restartRequired: boolean
}

GET /api/config/presets
Response: { presets: ConfigPreset[] }
```

### **Logging API**
```typescript
GET /api/logs/stream?service={service}&level={level}
Response: Server-Sent Events stream

GET /api/logs/search
Query: {
  query: string
  services?: string[]
  levels?: string[]
  startTime?: string
  endTime?: string
}
Response: {
  logs: LogEntry[]
  total: number
}
```

---

## 📡 ETAPA 2: Data Ingestion + Real-time Feeds

### **Market Data API**
```typescript
GET /api/market/ticker/{symbol}
Response: {
  symbol: string
  price: number
  change24h: number
  changePercent24h: number
  volume24h: number
  timestamp: string
}

GET /api/market/orderbook/{symbol}?depth={depth}
Response: {
  symbol: string
  bids: OrderLevel[]
  asks: OrderLevel[]
  timestamp: string
}

interface OrderLevel {
  price: number
  quantity: number
  total: number
}
```

### **Social Data API**
```typescript
GET /api/social/tweets?keywords={keywords}&limit={limit}
Response: {
  tweets: Tweet[]
  rateLimit: { remaining: number, resetTime: string }
}

interface Tweet {
  id: string
  text: string
  author: { username: string, followers: number, verified: boolean }
  timestamp: string
  metrics: { likes: number, retweets: number, replies: number }
  sentiment?: { score: number, confidence: number, label: string }
}

GET /api/social/sentiment/aggregate?timeframe={timeframe}
Response: {
  data: SentimentAggregation[]
}

interface SentimentAggregation {
  timestamp: string
  overall: number
  positive: number
  negative: number
  neutral: number
  volume: number
}
```

---

## 🧠 ETAPA 3: Feature Engineering + Technical Analysis

### **Features API**
```typescript
GET /api/features/current/{symbol}
Response: {
  symbol: string
  timestamp: string
  features: FeatureVector
}

interface FeatureVector {
  rsi_14: number
  macd: number
  bb_upper: number
  order_imbalance: number
  sentiment_score: number
  volatility_1h: number
  [key: string]: number
}

GET /api/features/importance/{symbol}
Response: {
  importance: FeatureImportance[]
}

interface FeatureImportance {
  feature: string
  importance: number
  rank: number
  category: string
}
```

### **Technical Analysis API**
```typescript
GET /api/technical/indicators/{symbol}?indicators={indicators}
Response: {
  indicators: TechnicalIndicatorData[]
}

interface TechnicalIndicatorData {
  name: string
  type: 'overlay' | 'oscillator' | 'volume'
  values: IndicatorValue[]
  signals: IndicatorSignal[]
}
```

---

## 🎯 ETAPA 4: Signal Generation + ML Predictions

### **Machine Learning API**
```typescript
GET /api/ml/models
Response: {
  models: MLModel[]
  activeModel: string
}

interface MLModel {
  id: string
  name: string
  type: 'lightgbm' | 'xgboost' | 'neural_network'
  status: 'training' | 'ready' | 'error'
  metrics: { accuracy: number, precision: number, recall: number }
  lastTrained: string
}

POST /api/ml/models/{modelId}/train
Body: {
  dataRange: { start: string, end: string }
  features: string[]
  parameters?: Record<string, any>
}
Response: { success: boolean, trainingId: string }
```

### **Signal Generation API**
```typescript
GET /api/signals/current/{symbol}
Response: {
  signal: TradingSignal | null
}

interface TradingSignal {
  id: string
  symbol: string
  timestamp: string
  action: 'BUY' | 'SELL' | 'HOLD'
  confidence: number
  price: number
  targetPrice?: number
  stopLoss?: number
  reasoning: string[]
  status: 'pending' | 'approved' | 'rejected' | 'executed'
}

POST /api/signals/{signalId}/approve
Response: { success: boolean }

POST /api/signals/{signalId}/reject
Body: { reason: string }
Response: { success: boolean }
```

---

## ⚠️ ETAPA 5: Risk Management + Portfolio Control

### **Portfolio API**
```typescript
GET /api/portfolio
Response: {
  portfolio: Portfolio
}

interface Portfolio {
  totalValue: number
  availableCash: number
  positions: Position[]
  dailyPnL: number
}

interface Position {
  id: string
  symbol: string
  quantity: number
  avgPrice: number
  currentPrice: number
  unrealizedPnL: number
}

GET /api/risk/metrics
Response: {
  metrics: RiskMetrics
  limits: RiskLimits
}

interface RiskMetrics {
  currentDrawdown: number
  maxDrawdown: number
  var95: number
  sharpeRatio: number
}
```

---

## 💰 ETAPA 6: Trading Execution + Order Management

### **Trading API**
```typescript
POST /api/trading/execute
Body: {
  symbol: string
  side: 'BUY' | 'SELL'
  quantity: number
  type: 'MARKET' | 'LIMIT'
  price?: number
}
Response: {
  success: boolean
  orderId: string
  executionReport: ExecutionReport
}

GET /api/trading/orders
Response: {
  orders: Order[]
}

interface Order {
  id: string
  symbol: string
  side: 'BUY' | 'SELL'
  quantity: number
  status: 'PENDING' | 'FILLED' | 'CANCELLED'
  timestamp: string
}
```

---

## 📊 ETAPA 7: Monitoring & Advanced Analytics

### **Analytics API**
```typescript
GET /api/analytics/performance?timeRange={timeRange}
Response: {
  metrics: PerformanceMetrics
}

interface PerformanceMetrics {
  totalReturn: number
  sharpeRatio: number
  maxDrawdown: number
  winRate: number
  trades: number
}

GET /api/analytics/backtest
Body: {
  strategy: string
  startDate: string
  endDate: string
  parameters: Record<string, any>
}
Response: {
  results: BacktestResult
}
```

---

## 🔄 ETAPA 8: Auto-Optimization

### **Optimization API**
```typescript
GET /api/optimization/schedule
Response: {
  schedule: TrainingSchedule
}

POST /api/optimization/hyperparameters/tune
Body: {
  modelId: string
  parameterSpace: Record<string, any>
  trials: number
}
Response: {
  success: boolean
  tuningId: string
}
```

---

## 🌐 WebSocket Events Summary

### **Connection Endpoints**
- `ws://localhost:8000/ws/system` - System events
- `ws://localhost:8000/ws/market` - Market data
- `ws://localhost:8000/ws/social` - Social data
- `ws://localhost:8000/ws/features` - Feature updates
- `ws://localhost:8000/ws/ml` - ML events
- `ws://localhost:8000/ws/signals` - Trading signals
- `ws://localhost:8000/ws/trading` - Trading execution
- `ws://localhost:8000/ws/risk` - Risk alerts

### **Common Event Types**
- `*.update` - Data updates
- `*.alert` - Alert notifications
- `*.status` - Status changes
- `*.error` - Error notifications

---

## 🔒 Error Handling

### **Standard Error Response**
```typescript
interface APIError {
  error: {
    code: string
    message: string
    details?: any
    timestamp: string
    requestId: string
  }
}

// HTTP Status Codes
200: Success
400: Bad Request
401: Unauthorized
403: Forbidden
404: Not Found
429: Rate Limited
500: Internal Server Error
503: Service Unavailable
```

### **Rate Limiting**
```typescript
// Headers included in responses
X-RateLimit-Limit: number
X-RateLimit-Remaining: number
X-RateLimit-Reset: timestamp
```

**Total endpoints**: ~80 across all stages
