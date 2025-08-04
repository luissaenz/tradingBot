# 🧩 Component Specifications - Next.js Interface

## 📋 Overview

Especificaciones detalladas de todos los componentes React que se implementarán en cada etapa del desarrollo, incluyendo props, estado, dependencias y responsabilidades.

---

## 🏗️ ETAPA 0: Interface Base

### **Layout Components**

#### `AppLayout.tsx`
```tsx
interface AppLayoutProps {
  children: React.ReactNode
}
// Responsabilidades: Estructura principal, sidebar, header, theme provider
```

#### `Sidebar.tsx`
```tsx
interface SidebarProps {
  currentPath: string
  isCollapsed?: boolean
  onToggle: () => void
}
// Responsabilidades: Navegación, indicadores de progreso, estado de servicios
```

#### `Header.tsx`
```tsx
interface HeaderProps {
  title: string
  systemStatus: 'setup' | 'developing' | 'active' | 'error'
  onEmergencyStop: () => void
}
// Responsabilidades: Título, estado del sistema, controles de emergencia
```

### **Setup Components**

#### `SetupProgress.tsx`
```tsx
interface SetupProgressProps {
  stages: StageProgress[]
  currentStage: number
}

interface StageProgress {
  name: string
  status: 'pending' | 'in-progress' | 'completed' | 'error'
  progress: number // 0-100
  estimatedTime?: string
}
// Responsabilidades: Progreso de setup, etapa actual, estimaciones
```

#### `ConfigurationPanel.tsx`
```tsx
interface ConfigurationPanelProps {
  configPath: string
  onConfigSelect: (path: string) => void
  onTestConnection: () => Promise<boolean>
  onVerifySetup: () => Promise<SetupResult>
}
// Responsabilidades: Selección de config, test de APIs, verificación
```

---

## 🔧 ETAPA 1: Shared Libraries + System Monitoring

### **System Health Components**

#### `ResourceMonitor.tsx`
```tsx
interface ResourceMonitorProps {
  refreshInterval?: number // default: 5000ms
  showAlerts?: boolean
}

interface SystemResources {
  cpu: { usage: number; cores: number; temperature?: number }
  memory: { used: number; total: number; percentage: number }
  disk: { used: number; total: number; percentage: number }
}
// Responsabilidades: Métricas de sistema, gráficos de tendencia, alertas
```

#### `ServiceStatus.tsx`
```tsx
interface ServiceStatusProps {
  services: ServiceInfo[]
  onRestartService: (serviceId: string) => Promise<void>
  onViewLogs: (serviceId: string) => void
}

interface ServiceInfo {
  id: string
  name: string
  status: 'running' | 'stopped' | 'error' | 'starting'
  uptime?: number
  memoryUsage?: number
  healthCheck?: { status: string; lastCheck: Date; responseTime?: number }
}
// Responsabilidades: Estado de microservicios, controles restart, health checks
```

#### `LogViewer.tsx`
```tsx
interface LogViewerProps {
  serviceId?: string
  level?: LogLevel[]
  maxLines?: number
  autoScroll?: boolean
  searchQuery?: string
}

interface LogEntry {
  timestamp: Date
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL'
  service: string
  message: string
  metadata?: Record<string, any>
}
// Responsabilidades: Stream de logs, filtrado, búsqueda, export
```

---

## 📡 ETAPA 2: Data Ingestion + Real-time Feeds

### **Market Data Components**

#### `PriceStream.tsx`
```tsx
interface PriceStreamProps {
  symbol: string
  showChange?: boolean
  showVolume?: boolean
  precision?: number
}

interface PriceData {
  symbol: string
  price: number
  change24h: number
  changePercent24h: number
  volume24h: number
  timestamp: Date
}
// Responsabilidades: Precio en tiempo real, indicadores visuales, animaciones
```

#### `OrderBookDepth.tsx`
```tsx
interface OrderBookDepthProps {
  symbol: string
  depth?: number
  showSpread?: boolean
  onPriceClick?: (price: number) => void
}

interface OrderBookData {
  symbol: string
  bids: OrderLevel[]
  asks: OrderLevel[]
  timestamp: Date
}

interface OrderLevel {
  price: number
  quantity: number
  total: number
}
// Responsabilidades: Order book en tiempo real, visualización de profundidad
```

#### `TwitterFeed.tsx`
```tsx
interface TwitterFeedProps {
  keywords: string[]
  maxTweets?: number
  showSentiment?: boolean
  onTweetClick?: (tweet: Tweet) => void
}

interface Tweet {
  id: string
  text: string
  author: { username: string; followers: number; verified: boolean }
  timestamp: Date
  metrics: { likes: number; retweets: number; replies: number }
  sentiment?: { score: number; confidence: number; label: string }
}
// Responsabilidades: Stream de tweets, filtrado, sentiment visual
```

---

## 🧠 ETAPA 3: Feature Engineering + Technical Analysis

### **Technical Analysis Components**

#### `IndicatorCharts.tsx`
```tsx
interface IndicatorChartsProps {
  symbol: string
  timeframe: string
  indicators: TechnicalIndicator[]
  height?: number
}

interface TechnicalIndicator {
  name: string
  type: 'overlay' | 'oscillator' | 'volume'
  parameters: Record<string, any>
  values: IndicatorValue[]
  alerts?: IndicatorAlert[]
}
// Responsabilidades: Gráficos de indicadores, overlays, alertas automáticas
```

#### `OrderFlowChart.tsx`
```tsx
interface OrderFlowChartProps {
  symbol: string
  timeframe: string
  showImbalances?: boolean
  showDelta?: boolean
}

interface OrderFlowData {
  timestamp: Date
  price: number
  buyVolume: number
  sellVolume: number
  delta: number
  imbalance: number
}
// Responsabilidades: Visualización de order flow, delta volume, imbalances
```

#### `FeatureMatrix.tsx`
```tsx
interface FeatureMatrixProps {
  features: FeatureVector[]
  showCorrelations?: boolean
  highlightImportant?: boolean
}

interface FeatureVector {
  timestamp: Date
  features: Record<string, number>
  target?: number
}
// Responsabilidades: Matrix de features, importance ranking, correlaciones
```

---

## 🎯 ETAPA 4: Signal Generation + ML Predictions

### **Machine Learning Components**

#### `ModelDashboard.tsx`
```tsx
interface ModelDashboardProps {
  models: MLModel[]
  activeModel: string
  onSwitchModel: (modelId: string) => void
}

interface MLModel {
  id: string
  name: string
  type: 'lightgbm' | 'xgboost' | 'neural_network'
  status: 'training' | 'ready' | 'error' | 'outdated'
  metrics: { accuracy: number; precision: number; recall: number; f1Score: number }
  lastTrained: Date
}
// Responsabilidades: Estado de modelos, métricas, comparación, switching
```

#### `SignalGenerator.tsx`
```tsx
interface SignalGeneratorProps {
  latestSignal?: TradingSignal
  signalHistory: TradingSignal[]
  onApproveSignal: (signalId: string) => void
  onRejectSignal: (signalId: string) => void
  autoApprove?: boolean
}

interface TradingSignal {
  id: string
  timestamp: Date
  action: 'BUY' | 'SELL' | 'HOLD'
  confidence: number
  price: number
  targetPrice?: number
  stopLoss?: number
  reasoning: string[]
  status: 'pending' | 'approved' | 'rejected' | 'executed'
}
// Responsabilidades: Display de señal, approval interface, reasoning
```

---

## ⚠️ ETAPA 5: Risk Management + Portfolio Control

### **Risk Management Components**

#### `RiskDashboard.tsx`
```tsx
interface RiskDashboardProps {
  portfolio: Portfolio
  riskMetrics: RiskMetrics
  limits: RiskLimits
  onUpdateLimits: (limits: RiskLimits) => void
}

interface Portfolio {
  totalValue: number
  availableCash: number
  positions: Position[]
  dailyPnL: number
}

interface RiskMetrics {
  currentDrawdown: number
  maxDrawdown: number
  var95: number
  sharpeRatio: number
}
// Responsabilidades: Overview de riesgo, métricas, límites configurables
```

#### `PositionManager.tsx`
```tsx
interface PositionManagerProps {
  positions: Position[]
  onClosePosition: (positionId: string) => void
  onModifyPosition: (positionId: string, modifications: any) => void
}

interface Position {
  id: string
  symbol: string
  quantity: number
  avgPrice: number
  currentPrice: number
  unrealizedPnL: number
  exposure: number
}
// Responsabilidades: Gestión de posiciones, cierre, modificación
```

---

## 💰 ETAPA 6: Trading Execution + Order Management

### **Trading Execution Components**

#### `TradeExecutor.tsx`
```tsx
interface TradeExecutorProps {
  pendingSignal?: TradingSignal
  executionStatus: ExecutionStatus
  onExecute: () => void
  onCancel: () => void
}

interface ExecutionStatus {
  status: 'idle' | 'executing' | 'completed' | 'failed'
  orderId?: string
  fillPrice?: number
  slippage?: number
  latency?: number
}
// Responsabilidades: Ejecución de trades, status, métricas de ejecución
```

#### `OrderManager.tsx`
```tsx
interface OrderManagerProps {
  orders: Order[]
  onCancelOrder: (orderId: string) => void
  onModifyOrder: (orderId: string, modifications: any) => void
}

interface Order {
  id: string
  symbol: string
  side: 'BUY' | 'SELL'
  type: 'MARKET' | 'LIMIT' | 'STOP'
  quantity: number
  price?: number
  status: 'PENDING' | 'FILLED' | 'CANCELLED' | 'REJECTED'
  timestamp: Date
}
// Responsabilidades: Gestión de órdenes, cancelación, modificación
```

---

## 📊 ETAPA 7: Monitoring & Advanced Analytics

### **Analytics Components**

#### `PerformanceDashboard.tsx`
```tsx
interface PerformanceDashboardProps {
  timeRange: TimeRange
  metrics: PerformanceMetrics
  benchmarks?: Benchmark[]
}

interface PerformanceMetrics {
  totalReturn: number
  sharpeRatio: number
  maxDrawdown: number
  winRate: number
  profitFactor: number
  trades: number
}
// Responsabilidades: Métricas de performance, benchmarks, análisis temporal
```

#### `BacktestingInterface.tsx`
```tsx
interface BacktestingInterfaceProps {
  strategies: Strategy[]
  onRunBacktest: (config: BacktestConfig) => void
  results?: BacktestResult[]
}

interface BacktestConfig {
  strategy: string
  startDate: Date
  endDate: Date
  initialCapital: number
  parameters: Record<string, any>
}
// Responsabilidades: Interface de backtesting, configuración, resultados
```

---

## 🔄 ETAPA 8: Auto-Optimization + Advanced Features

### **Optimization Components**

#### `AutoTrainer.tsx`
```tsx
interface AutoTrainerProps {
  schedule: TrainingSchedule
  onUpdateSchedule: (schedule: TrainingSchedule) => void
  trainingHistory: TrainingRun[]
}

interface TrainingSchedule {
  frequency: 'hourly' | 'daily' | 'weekly'
  conditions: TrainingCondition[]
  enabled: boolean
}
// Responsabilidades: Entrenamiento automático, scheduling, historial
```

#### `HyperparameterTuning.tsx`
```tsx
interface HyperparameterTuningProps {
  currentParams: Record<string, any>
  tuningResults: TuningResult[]
  onStartTuning: (config: TuningConfig) => void
}

interface TuningResult {
  parameters: Record<string, any>
  score: number
  metrics: Record<string, number>
  timestamp: Date
}
// Responsabilidades: Tuning de hiperparámetros, resultados, optimización
```

---

## 🎯 Patrones de Diseño Comunes

### **Props Patterns**
- `onAction` callbacks para interacciones
- `show*` boolean props para features opcionales
- `*Config` objects para configuraciones complejas
- `timeRange` para componentes temporales

### **State Patterns**
- Zustand stores por dominio (trading, system, ml, etc.)
- React Query para data fetching
- Local state solo para UI state

### **Styling Patterns**
- shadcn/ui components como base
- Tailwind CSS para customización
- CSS variables para theming
- Responsive design mobile-first

### **Performance Patterns**
- React.memo para componentes pesados
- useMemo para cálculos complejos
- Virtualization para listas largas
- Debouncing para inputs de búsqueda

**Total estimado**: ~150 componentes React across all stages
