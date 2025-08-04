# 🗄️ State Management - Zustand Stores & Global State

## 📋 Overview

Arquitectura de estado global para la interfaz Next.js usando Zustand, React Query y patrones de estado optimizados para trading en tiempo real.

---

## 🏗️ Arquitectura de Estado

### **Store Structure**
```
stores/
├── system/           # System health & configuration
├── market/           # Market data & real-time feeds
├── features/         # Feature engineering & technical analysis
├── ml/              # Machine learning & models
├── signals/         # Trading signals & predictions
├── risk/            # Risk management & portfolio
├── trading/         # Trading execution & orders
├── analytics/       # Performance analytics & reports
├── ui/              # UI state & preferences
└── auth/            # Authentication & user management
```

### **State Flow Pattern**
```
WebSocket/API → Store → Component → UI Update
     ↑              ↓
     └── Actions ←──┘
```

---

## 🔧 ETAPA 1: System Stores

### **System Store**
```typescript
// stores/system/systemStore.ts
interface SystemState {
  status: 'setup' | 'developing' | 'active' | 'error'
  uptime: number
  resources: SystemResources | null
  services: ServiceInfo[]
  serviceAlerts: SystemAlert[]
  
  // Actions
  updateResources: (resources: SystemResources) => void
  updateServiceStatus: (serviceId: string, status: ServiceInfo) => void
  addAlert: (alert: SystemAlert) => void
  dismissAlert: (alertId: string) => void
  restartService: (serviceId: string) => Promise<void>
}

const useSystemStore = create<SystemState>()(
  subscribeWithSelector(
    devtools((set, get) => ({
      status: 'setup',
      uptime: 0,
      resources: null,
      services: [],
      serviceAlerts: [],
      
      updateResources: (resources) => set((state) => ({
        resources,
        resourceHistory: [resources, ...state.resourceHistory.slice(0, 99)]
      })),
      
      restartService: async (serviceId) => {
        const response = await fetch(`/api/system/services/${serviceId}/restart`, {
          method: 'POST'
        })
        if (!response.ok) throw new Error('Failed to restart service')
      }
    }), { name: 'system-store' })
  )
)
```

### **Configuration Store**
```typescript
// stores/system/configStore.ts
interface ConfigState {
  config: Record<string, any>
  originalConfig: Record<string, any>
  hasChanges: boolean
  presets: ConfigPreset[]
  validationErrors: ValidationError[]
  
  // Actions
  updateConfig: (path: string, value: any) => void
  resetConfig: () => void
  saveConfig: () => Promise<void>
  loadPreset: (presetId: string) => void
  validateConfig: () => Promise<void>
}

const useConfigStore = create<ConfigState>()(
  devtools((set, get) => ({
    config: {},
    originalConfig: {},
    hasChanges: false,
    presets: [],
    validationErrors: [],
    
    updateConfig: (path, value) => set((state) => {
      const newConfig = { ...state.config }
      setNestedValue(newConfig, path, value)
      return {
        config: newConfig,
        hasChanges: !deepEqual(newConfig, state.originalConfig)
      }
    }),
    
    saveConfig: async () => {
      const { config } = get()
      const response = await fetch('/api/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config })
      })
      if (!response.ok) throw new Error('Failed to save config')
      set({ originalConfig: { ...config }, hasChanges: false })
    }
  }), { name: 'config-store' })
)
```

---

## 📡 ETAPA 2: Market Data Stores

### **Market Store**
```typescript
// stores/market/marketStore.ts
interface MarketState {
  currentPrice: Record<string, PriceData>
  orderBooks: Record<string, OrderBookData>
  recentTrades: Record<string, Trade[]>
  connectionStatus: Record<string, ConnectionStatus>
  
  // Actions
  updatePrice: (symbol: string, price: PriceData) => void
  updateOrderBook: (symbol: string, orderBook: OrderBookData) => void
  addTrade: (symbol: string, trade: Trade) => void
}

const useMarketStore = create<MarketState>()(
  subscribeWithSelector(
    devtools((set, get) => ({
      currentPrice: {},
      orderBooks: {},
      recentTrades: {},
      connectionStatus: {},
      
      updatePrice: (symbol, price) => set((state) => ({
        currentPrice: { ...state.currentPrice, [symbol]: price }
      })),
      
      addTrade: (symbol, trade) => set((state) => ({
        recentTrades: {
          ...state.recentTrades,
          [symbol]: [trade, ...(state.recentTrades[symbol] || []).slice(0, 99)]
        }
      }))
    }), { name: 'market-store' })
  )
)
```

### **Social Store**
```typescript
// stores/market/socialStore.ts
interface SocialState {
  tweets: Tweet[]
  currentSentiment: Record<string, SentimentData>
  hashtagTrends: HashtagTrend[]
  activeKeywords: string[]
  
  // Actions
  addTweet: (tweet: Tweet) => void
  updateSentiment: (symbol: string, sentiment: SentimentData) => void
  updateTrends: (trends: HashtagTrend[]) => void
}

const useSocialStore = create<SocialState>()(
  subscribeWithSelector(
    devtools((set, get) => ({
      tweets: [],
      currentSentiment: {},
      hashtagTrends: [],
      activeKeywords: ['bitcoin', 'btc', 'crypto'],
      
      addTweet: (tweet) => set((state) => ({
        tweets: [tweet, ...state.tweets.slice(0, 199)]
      })),
      
      updateSentiment: (symbol, sentiment) => set((state) => ({
        currentSentiment: { ...state.currentSentiment, [symbol]: sentiment }
      }))
    }), { name: 'social-store' })
  )
)
```

---

## 🧠 ETAPA 3: Features Store

### **Features Store**
```typescript
// stores/features/featuresStore.ts
interface FeaturesState {
  currentFeatures: Record<string, FeatureVector>
  featureImportance: Record<string, FeatureImportance[]>
  indicators: Record<string, TechnicalIndicatorData[]>
  customIndicators: CustomIndicator[]
  
  // Actions
  updateFeatures: (symbol: string, features: FeatureVector) => void
  updateIndicators: (symbol: string, indicators: TechnicalIndicatorData[]) => void
  addCustomIndicator: (indicator: CustomIndicator) => void
}

const useFeaturesStore = create<FeaturesState>()(
  subscribeWithSelector(
    devtools((set, get) => ({
      currentFeatures: {},
      featureImportance: {},
      indicators: {},
      customIndicators: [],
      
      updateFeatures: (symbol, features) => set((state) => ({
        currentFeatures: { ...state.currentFeatures, [symbol]: features }
      })),
      
      addCustomIndicator: (indicator) => set((state) => ({
        customIndicators: [...state.customIndicators, indicator]
      }))
    }), { name: 'features-store' })
  )
)
```

---

## 🎯 ETAPA 4: ML & Signals Stores

### **ML Store**
```typescript
// stores/ml/mlStore.ts
interface MLState {
  models: MLModel[]
  activeModel: string | null
  trainingStatus: Record<string, TrainingProgress>
  
  // Actions
  addModel: (model: MLModel) => void
  setActiveModel: (modelId: string) => void
  startTraining: (modelId: string, config: TrainingConfig) => Promise<string>
}

const useMLStore = create<MLState>()(
  subscribeWithSelector(
    devtools((set, get) => ({
      models: [],
      activeModel: null,
      trainingStatus: {},
      
      setActiveModel: (modelId) => set({ activeModel: modelId }),
      
      startTraining: async (modelId, config) => {
        const response = await fetch(`/api/ml/models/${modelId}/train`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(config)
        })
        const result = await response.json()
        return result.trainingId
      }
    }), { name: 'ml-store' })
  )
)
```

### **Signals Store**
```typescript
// stores/signals/signalsStore.ts
interface SignalsState {
  currentSignals: Record<string, TradingSignal>
  signalHistory: TradingSignal[]
  predictions: Record<string, Prediction[]>
  autoApprove: boolean
  confidenceThreshold: number
  
  // Actions
  addSignal: (signal: TradingSignal) => void
  approveSignal: (signalId: string) => Promise<void>
  rejectSignal: (signalId: string, reason: string) => Promise<void>
}

const useSignalsStore = create<SignalsState>()(
  subscribeWithSelector(
    devtools((set, get) => ({
      currentSignals: {},
      signalHistory: [],
      predictions: {},
      autoApprove: false,
      confidenceThreshold: 75,
      
      addSignal: (signal) => set((state) => ({
        currentSignals: { ...state.currentSignals, [signal.symbol]: signal },
        signalHistory: [signal, ...state.signalHistory.slice(0, 199)]
      })),
      
      approveSignal: async (signalId) => {
        const response = await fetch(`/api/signals/${signalId}/approve`, {
          method: 'POST'
        })
        if (!response.ok) throw new Error('Failed to approve signal')
      }
    }), { name: 'signals-store' })
  )
)
```

---

## ⚠️ ETAPA 5: Risk & Portfolio Stores

### **Portfolio Store**
```typescript
// stores/risk/portfolioStore.ts
interface PortfolioState {
  portfolio: Portfolio | null
  positions: Position[]
  dailyPnL: number
  
  // Actions
  updatePortfolio: (portfolio: Portfolio) => void
  closePosition: (positionId: string) => Promise<void>
}

const usePortfolioStore = create<PortfolioState>()(
  devtools((set, get) => ({
    portfolio: null,
    positions: [],
    dailyPnL: 0,
    
    updatePortfolio: (portfolio) => set({ 
      portfolio,
      positions: portfolio.positions,
      dailyPnL: portfolio.dailyPnL
    }),
    
    closePosition: async (positionId) => {
      const response = await fetch(`/api/positions/${positionId}/close`, {
        method: 'POST'
      })
      if (!response.ok) throw new Error('Failed to close position')
    }
  }), { name: 'portfolio-store' })
)
```

---

## 💰 ETAPA 6: Trading Store

### **Trading Store**
```typescript
// stores/trading/tradingStore.ts
interface TradingState {
  orders: Order[]
  executionStatus: Record<string, ExecutionStatus>
  tradingEnabled: boolean
  
  // Actions
  executeSignal: (signal: TradingSignal) => Promise<string>
  cancelOrder: (orderId: string) => Promise<void>
  setTradingEnabled: (enabled: boolean) => void
}

const useTradingStore = create<TradingState>()(
  devtools((set, get) => ({
    orders: [],
    executionStatus: {},
    tradingEnabled: false,
    
    executeSignal: async (signal) => {
      const response = await fetch('/api/trading/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: signal.symbol,
          side: signal.action,
          quantity: signal.positionSize
        })
      })
      const result = await response.json()
      return result.orderId
    },
    
    setTradingEnabled: (enabled) => set({ tradingEnabled: enabled })
  }), { name: 'trading-store' })
)
```

---

## 🎨 UI Store

### **UI Store**
```typescript
// stores/ui/uiStore.ts
interface UIState {
  theme: 'light' | 'dark' | 'system'
  sidebarCollapsed: boolean
  activeTab: string
  modals: Record<string, boolean>
  notifications: Notification[]
  
  // Actions
  setTheme: (theme: 'light' | 'dark' | 'system') => void
  toggleSidebar: () => void
  setActiveTab: (tab: string) => void
  addNotification: (notification: Notification) => void
}

const useUIStore = create<UIState>()(
  persist(
    devtools((set, get) => ({
      theme: 'dark',
      sidebarCollapsed: false,
      activeTab: 'dashboard',
      modals: {},
      notifications: [],
      
      setTheme: (theme) => set({ theme }),
      toggleSidebar: () => set((state) => ({
        sidebarCollapsed: !state.sidebarCollapsed
      })),
      addNotification: (notification) => set((state) => ({
        notifications: [notification, ...state.notifications.slice(0, 9)]
      }))
    }), { name: 'ui-store' }),
    { name: 'ui-store-persist' }
  )
)
```

---

## 🔄 WebSocket Integration

### **WebSocket Hook**
```typescript
// hooks/useWebSocket.ts
export function useWebSocket() {
  const updatePrice = useMarketStore(state => state.updatePrice)
  const addTrade = useMarketStore(state => state.addTrade)
  const addSignal = useSignalsStore(state => state.addSignal)
  const updateResources = useSystemStore(state => state.updateResources)

  useEffect(() => {
    const sockets = {
      market: io('ws://localhost:8000/ws/market'),
      system: io('ws://localhost:8000/ws/system'),
      signals: io('ws://localhost:8000/ws/signals')
    }

    // Market events
    sockets.market.on('market.ticker.update', updatePrice)
    sockets.market.on('market.trade', addTrade)
    
    // System events
    sockets.system.on('system.resources.update', updateResources)
    
    // Signal events
    sockets.signals.on('signal.generated', addSignal)

    return () => {
      Object.values(sockets).forEach(socket => socket.disconnect())
    }
  }, [])
}
```

---

## 📊 React Query Integration

### **Query Client Setup**
```typescript
// lib/queryClient.ts
import { QueryClient } from '@tanstack/react-query'

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30 * 1000, // 30 seconds
      cacheTime: 5 * 60 * 1000, // 5 minutes
      refetchOnWindowFocus: false,
      retry: 3
    }
  }
})
```

### **Custom Hooks**
```typescript
// hooks/useMarketData.ts
export function useMarketData(symbol: string) {
  return useQuery({
    queryKey: ['market', 'ticker', symbol],
    queryFn: () => fetch(`/api/market/ticker/${symbol}`).then(r => r.json()),
    refetchInterval: 1000 // 1 second
  })
}

// hooks/useSignalHistory.ts
export function useSignalHistory(symbol: string, timeRange: TimeRange) {
  return useQuery({
    queryKey: ['signals', 'history', symbol, timeRange],
    queryFn: () => fetch(`/api/signals/history/${symbol}?start=${timeRange.start}&end=${timeRange.end}`).then(r => r.json()),
    staleTime: 60 * 1000 // 1 minute
  })
}
```

---

## 🎯 Store Patterns & Best Practices

### **Subscription Pattern**
```typescript
// Subscribe to specific state changes
useSystemStore.subscribe(
  (state) => state.resources,
  (resources) => {
    if (resources?.cpu.usage > 90) {
      // Trigger high CPU alert
    }
  }
)
```

### **Computed Values**
```typescript
// Derived state with selectors
const totalPnL = usePortfolioStore(
  (state) => state.positions.reduce((sum, pos) => sum + pos.unrealizedPnL, 0)
)
```

### **Optimistic Updates**
```typescript
// Update UI immediately, rollback on error
const approveSignal = async (signalId: string) => {
  // Optimistic update
  updateSignal(signalId, { status: 'approved' })
  
  try {
    await api.approveSignal(signalId)
  } catch (error) {
    // Rollback on error
    updateSignal(signalId, { status: 'pending' })
    throw error
  }
}
```

**Total stores**: ~12 across all stages  
**State management**: Zustand + React Query + WebSocket integration
