// src/web/components/ConfigurationManager.tsx
import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { 
  Settings, 
  TrendingUp, 
  History, 
  AlertTriangle, 
  CheckCircle, 
  RefreshCw,
  Save,
  Undo2,
  TestTube,
  BarChart3,
  Database
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface Parameter {
  module_name: string;
  parameter_name: string;
  current_value: any;
  parameter_type: string;
  min_value?: number;
  max_value?: number;
  description: string;
  optimization_enabled: boolean;
  last_updated: string;
  updated_by: string;
}

interface ParameterChange {
  id: number;
  old_value: any;
  new_value: any;
  changed_by: string;
  changed_at: string;
  change_reason: string;
  performance_before?: number;
  performance_after?: number;
}

interface ABTest {
  id: number;
  test_name: string;
  module_name: string;
  parameter_name: string;
  control_value: any;
  test_value: any;
  traffic_split: number;
  status: string;
  start_date: string;
  end_date?: string;
  winner?: string;
  confidence_level?: number;
}

const ConfigurationManager: React.FC = () => {
  // Estados principales
  const [modules, setModules] = useState<string[]>([]);
  const [selectedModule, setSelectedModule] = useState<string>('');
  const [parameters, setParameters] = useState<Parameter[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Estados para modales y formularios
  const [editingParameter, setEditingParameter] = useState<Parameter | null>(null);
  const [parameterHistory, setParameterHistory] = useState<ParameterChange[]>([]);
  const [abTests, setAbTests] = useState<ABTest[]>([]);
  const [showHistoryModal, setShowHistoryModal] = useState(false);
  const [showABTestModal, setShowABTestModal] = useState(false);
  const [showOptimizationModal, setShowOptimizationModal] = useState(false);

  // Estados para A/B Testing
  const [newABTest, setNewABTest] = useState({
    test_name: '',
    control_value: '',
    test_value: '',
    traffic_split: 50,
    duration_hours: 24
  });

  // Estados para optimización
  const [optimizationStatus, setOptimizationStatus] = useState<any>(null);
  const [optimizationResults, setOptimizationResults] = useState<any>(null);

  // WebSocket para notificaciones en tiempo real
  const [ws, setWs] = useState<WebSocket | null>(null);

  // Cargar módulos disponibles
  useEffect(() => {
    loadModules();
    setupWebSocket();
    
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []);

  // Cargar parámetros cuando cambia el módulo seleccionado
  useEffect(() => {
    if (selectedModule) {
      loadModuleParameters(selectedModule);
      loadABTests();
    }
  }, [selectedModule]);

  const setupWebSocket = () => {
    const websocket = new WebSocket('ws://localhost:8000/api/config/ws/notifications');
    
    websocket.onopen = () => {
      console.log('WebSocket connected');
      setWs(websocket);
    };
    
    websocket.onmessage = (event) => {
      const notification = JSON.parse(event.data);
      handleRealtimeNotification(notification);
    };
    
    websocket.onclose = () => {
      console.log('WebSocket disconnected');
      // Reconectar después de 5 segundos
      setTimeout(setupWebSocket, 5000);
    };
    
    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  };

  const handleRealtimeNotification = (notification: any) => {
    switch (notification.type) {
      case 'parameter_updated':
        // Actualizar parámetro en la lista
        setParameters(prev => prev.map(p => 
          p.module_name === notification.module_name && 
          p.parameter_name === notification.parameter_name
            ? { ...p, current_value: notification.new_value, last_updated: notification.timestamp }
            : p
        ));
        setSuccess(`Parameter ${notification.parameter_name} updated to ${notification.new_value}`);
        break;
        
      case 'optimization_completed':
        setOptimizationResults(notification.results);
        setSuccess(`Optimization completed for ${notification.module_name}`);
        break;
        
      case 'ab_test_result':
        setSuccess(`A/B test ${notification.test_name} completed. Winner: ${notification.winner}`);
        break;
    }
  };

  const loadModules = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/config/modules');
      const data = await response.json();
      
      if (data.status === 'success') {
        setModules(data.modules);
        if (data.modules.length > 0 && !selectedModule) {
          setSelectedModule(data.modules[0]);
        }
      }
    } catch (err) {
      setError('Failed to load modules');
    } finally {
      setLoading(false);
    }
  };

  const loadModuleParameters = async (moduleName: string) => {
    try {
      setLoading(true);
      const response = await fetch(`/api/config/parameters/${moduleName}`);
      const data = await response.json();
      
      if (data.status === 'success') {
        setParameters(data.parameters);
      }
    } catch (err) {
      setError('Failed to load parameters');
    } finally {
      setLoading(false);
    }
  };

  const loadABTests = async () => {
    try {
      const response = await fetch('/api/config/ab-tests');
      const data = await response.json();
      
      if (data.status === 'success') {
        setAbTests(data.ab_tests);
      }
    } catch (err) {
      console.error('Failed to load A/B tests');
    }
  };

  const updateParameter = async (parameter: Parameter, newValue: any) => {
    try {
      setLoading(true);
      const response = await fetch(
        `/api/config/parameters/${parameter.module_name}/${parameter.parameter_name}`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            value: newValue,
            changed_by: 'web_user',
            reason: 'manual'
          })
        }
      );
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setSuccess(`Parameter ${parameter.parameter_name} updated successfully`);
        await loadModuleParameters(selectedModule);
      } else {
        setError(data.message || 'Failed to update parameter');
      }
    } catch (err) {
      setError('Failed to update parameter');
    } finally {
      setLoading(false);
    }
  };

  const loadParameterHistory = async (parameter: Parameter) => {
    try {
      const response = await fetch(
        `/api/config/history/${parameter.module_name}/${parameter.parameter_name}`
      );
      const data = await response.json();
      
      if (data.status === 'success') {
        setParameterHistory(data.history);
        setShowHistoryModal(true);
      }
    } catch (err) {
      setError('Failed to load parameter history');
    }
  };

  const rollbackParameter = async (parameter: Parameter, targetTimestamp: string) => {
    try {
      setLoading(true);
      const response = await fetch(
        `/api/config/rollback/${parameter.module_name}/${parameter.parameter_name}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            target_timestamp: targetTimestamp,
            rolled_back_by: 'web_user'
          })
        }
      );
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setSuccess('Parameter rolled back successfully');
        await loadModuleParameters(selectedModule);
        setShowHistoryModal(false);
      } else {
        setError('Failed to rollback parameter');
      }
    } catch (err) {
      setError('Failed to rollback parameter');
    } finally {
      setLoading(false);
    }
  };

  const createABTest = async () => {
    if (!editingParameter) return;
    
    try {
      setLoading(true);
      const response = await fetch('/api/config/ab-tests', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          test_name: newABTest.test_name,
          module_name: editingParameter.module_name,
          parameter_name: editingParameter.parameter_name,
          control_value: editingParameter.current_value,
          test_value: parseValue(newABTest.test_value, editingParameter.parameter_type),
          traffic_split: newABTest.traffic_split / 100,
          duration_hours: newABTest.duration_hours,
          created_by: 'web_user'
        })
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setSuccess('A/B test created successfully');
        setShowABTestModal(false);
        await loadABTests();
      } else {
        setError('Failed to create A/B test');
      }
    } catch (err) {
      setError('Failed to create A/B test');
    } finally {
      setLoading(false);
    }
  };

  const optimizeParameters = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/config/optimize/${selectedModule}`, {
        method: 'POST'
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setSuccess('Parameter optimization started');
        setShowOptimizationModal(true);
        
        // Polling para obtener resultados
        const pollOptimization = setInterval(async () => {
          try {
            const statusResponse = await fetch(`/api/config/optimize/${selectedModule}/status`);
            const statusData = await statusResponse.json();
            
            if (statusData.optimization_status?.status === 'completed') {
              setOptimizationResults(statusData.optimization_status.results);
              clearInterval(pollOptimization);
            }
          } catch (err) {
            console.error('Error polling optimization status');
          }
        }, 2000);
        
        // Limpiar polling después de 5 minutos
        setTimeout(() => clearInterval(pollOptimization), 300000);
      }
    } catch (err) {
      setError('Failed to start optimization');
    } finally {
      setLoading(false);
    }
  };

  const parseValue = (value: string, type: string): any => {
    switch (type) {
      case 'float':
        return parseFloat(value);
      case 'int':
        return parseInt(value);
      case 'boolean':
        return value === 'true';
      case 'array':
        return JSON.parse(value);
      case 'object':
        return JSON.parse(value);
      default:
        return value;
    }
  };

  const formatValue = (value: any, type: string): string => {
    if (type === 'array' || type === 'object') {
      return JSON.stringify(value, null, 2);
    }
    return String(value);
  };

  const renderParameterInput = (parameter: Parameter) => {
    const { parameter_type, current_value, min_value, max_value } = parameter;
    
    switch (parameter_type) {
      case 'boolean':
        return (
          <Switch
            checked={current_value}
            onCheckedChange={(checked) => updateParameter(parameter, checked)}
          />
        );
        
      case 'float':
      case 'int':
        if (min_value !== undefined && max_value !== undefined) {
          return (
            <div className="space-y-2">
              <Slider
                value={[current_value]}
                onValueChange={([value]) => updateParameter(parameter, value)}
                min={min_value}
                max={max_value}
                step={parameter_type === 'float' ? 0.01 : 1}
                className="w-full"
              />
              <div className="flex justify-between text-sm text-gray-500">
                <span>{min_value}</span>
                <span className="font-medium">{current_value}</span>
                <span>{max_value}</span>
              </div>
            </div>
          );
        }
        return (
          <Input
            type="number"
            value={current_value}
            onChange={(e) => {
              const value = parameter_type === 'float' 
                ? parseFloat(e.target.value) 
                : parseInt(e.target.value);
              updateParameter(parameter, value);
            }}
            step={parameter_type === 'float' ? 0.01 : 1}
          />
        );
        
      case 'array':
      case 'object':
        return (
          <textarea
            className="w-full h-24 p-2 border rounded font-mono text-sm"
            value={formatValue(current_value, parameter_type)}
            onChange={(e) => {
              try {
                const parsed = JSON.parse(e.target.value);
                updateParameter(parameter, parsed);
              } catch (err) {
                // Mostrar error de parsing
              }
            }}
          />
        );
        
      default:
        return (
          <Input
            value={current_value}
            onChange={(e) => updateParameter(parameter, e.target.value)}
          />
        );
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Settings className="h-8 w-8" />
            Configuration Manager
          </h1>
          <p className="text-gray-600 mt-1">
            Manage trading bot parameters with real-time updates and A/B testing
          </p>
        </div>
        
        <div className="flex gap-2">
          <Button
            onClick={() => loadModuleParameters(selectedModule)}
            variant="outline"
            size="sm"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          
          <Button
            onClick={optimizeParameters}
            disabled={!selectedModule || loading}
            size="sm"
          >
            <TrendingUp className="h-4 w-4 mr-2" />
            Optimize
          </Button>
        </div>
      </div>

      {/* Alerts */}
      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      
      {success && (
        <Alert>
          <CheckCircle className="h-4 w-4" />
          <AlertDescription>{success}</AlertDescription>
        </Alert>
      )}

      {/* Module Selector */}
      <Card>
        <CardHeader>
          <CardTitle>Select Module</CardTitle>
        </CardHeader>
        <CardContent>
          <Select value={selectedModule} onValueChange={setSelectedModule}>
            <SelectTrigger>
              <SelectValue placeholder="Select a module" />
            </SelectTrigger>
            <SelectContent>
              {modules.map(module => (
                <SelectItem key={module} value={module}>
                  {module}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </CardContent>
      </Card>

      {/* Main Content */}
      {selectedModule && (
        <Tabs defaultValue="parameters" className="space-y-4">
          <TabsList>
            <TabsTrigger value="parameters">Parameters</TabsTrigger>
            <TabsTrigger value="ab-tests">A/B Tests</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
          </TabsList>

          {/* Parameters Tab */}
          <TabsContent value="parameters" className="space-y-4">
            <div className="grid gap-4">
              {parameters.map((parameter) => (
                <Card key={`${parameter.module_name}-${parameter.parameter_name}`}>
                  <CardHeader className="pb-3">
                    <div className="flex justify-between items-start">
                      <div>
                        <CardTitle className="text-lg">
                          {parameter.parameter_name}
                        </CardTitle>
                        <p className="text-sm text-gray-600 mt-1">
                          {parameter.description}
                        </p>
                      </div>
                      
                      <div className="flex gap-2">
                        {parameter.optimization_enabled && (
                          <Badge variant="secondary">
                            <TrendingUp className="h-3 w-3 mr-1" />
                            Auto-Opt
                          </Badge>
                        )}
                        
                        <Button
                          onClick={() => loadParameterHistory(parameter)}
                          variant="outline"
                          size="sm"
                        >
                          <History className="h-4 w-4" />
                        </Button>
                        
                        <Button
                          onClick={() => {
                            setEditingParameter(parameter);
                            setShowABTestModal(true);
                          }}
                          variant="outline"
                          size="sm"
                        >
                          <TestTube className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  
                  <CardContent>
                    <div className="space-y-3">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-500">Type:</span>
                        <Badge variant="outline">{parameter.parameter_type}</Badge>
                      </div>
                      
                      <div className="space-y-2">
                        <Label>Current Value</Label>
                        {renderParameterInput(parameter)}
                      </div>
                      
                      <div className="flex justify-between text-xs text-gray-500">
                        <span>Updated by: {parameter.updated_by}</span>
                        <span>{new Date(parameter.last_updated).toLocaleString()}</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* A/B Tests Tab */}
          <TabsContent value="ab-tests" className="space-y-4">
            <div className="grid gap-4">
              {abTests.map((test) => (
                <Card key={test.id}>
                  <CardHeader>
                    <div className="flex justify-between items-start">
                      <div>
                        <CardTitle>{test.test_name}</CardTitle>
                        <p className="text-sm text-gray-600">
                          {test.module_name}.{test.parameter_name}
                        </p>
                      </div>
                      
                      <Badge 
                        variant={test.status === 'active' ? 'default' : 'secondary'}
                      >
                        {test.status}
                      </Badge>
                    </div>
                  </CardHeader>
                  
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label>Control Value</Label>
                        <div className="font-mono text-sm bg-gray-100 p-2 rounded">
                          {formatValue(test.control_value, 'string')}
                        </div>
                      </div>
                      
                      <div>
                        <Label>Test Value</Label>
                        <div className="font-mono text-sm bg-gray-100 p-2 rounded">
                          {formatValue(test.test_value, 'string')}
                        </div>
                      </div>
                    </div>
                    
                    <div className="mt-4 flex justify-between items-center">
                      <div className="text-sm text-gray-600">
                        Traffic Split: {test.traffic_split * 100}%
                      </div>
                      
                      {test.winner && (
                        <Badge variant="default">
                          Winner: {test.winner}
                        </Badge>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Performance Tab */}
          <TabsContent value="performance" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Module Performance Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={[]}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis />
                      <Tooltip />
                      <Line 
                        type="monotone" 
                        dataKey="performance" 
                        stroke="#8884d8" 
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}

      {/* Modals */}
      
      {/* Parameter History Modal */}
      <Dialog open={showHistoryModal} onOpenChange={setShowHistoryModal}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle>Parameter History</DialogTitle>
          </DialogHeader>
          
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {parameterHistory.map((change) => (
              <div key={change.id} className="border rounded p-3">
                <div className="flex justify-between items-start">
                  <div>
                    <div className="text-sm font-medium">
                      {formatValue(change.old_value, 'string')} → {formatValue(change.new_value, 'string')}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      By {change.changed_by} • {change.change_reason}
                    </div>
                  </div>
                  
                  <div className="flex gap-2">
                    <Button
                      onClick={() => rollbackParameter(editingParameter!, change.changed_at)}
                      variant="outline"
                      size="sm"
                    >
                      <Undo2 className="h-3 w-3 mr-1" />
                      Rollback
                    </Button>
                  </div>
                </div>
                
                <div className="text-xs text-gray-400 mt-2">
                  {new Date(change.changed_at).toLocaleString()}
                </div>
              </div>
            ))}
          </div>
        </DialogContent>
      </Dialog>

      {/* A/B Test Creation Modal */}
      <Dialog open={showABTestModal} onOpenChange={setShowABTestModal}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create A/B Test</DialogTitle>
          </DialogHeader>
          
          <div className="space-y-4">
            <div>
              <Label>Test Name</Label>
              <Input
                value={newABTest.test_name}
                onChange={(e) => setNewABTest(prev => ({ ...prev, test_name: e.target.value }))}
                placeholder="Enter test name"
              />
            </div>
            
            <div>
              <Label>Test Value</Label>
              <Input
                value={newABTest.test_value}
                onChange={(e) => setNewABTest(prev => ({ ...prev, test_value: e.target.value }))}
                placeholder="Enter test value"
              />
            </div>
            
            <div>
              <Label>Traffic Split (%)</Label>
              <Slider
                value={[newABTest.traffic_split]}
                onValueChange={([value]) => setNewABTest(prev => ({ ...prev, traffic_split: value }))}
                min={10}
                max={90}
                step={5}
              />
              <div className="text-sm text-gray-500 mt-1">
                {newABTest.traffic_split}% will see the test value
              </div>
            </div>
            
            <div>
              <Label>Duration (hours)</Label>
              <Input
                type="number"
                value={newABTest.duration_hours}
                onChange={(e) => setNewABTest(prev => ({ ...prev, duration_hours: parseInt(e.target.value) }))}
                min={1}
                max={168}
              />
            </div>
            
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setShowABTestModal(false)}>
                Cancel
              </Button>
              <Button onClick={createABTest} disabled={loading}>
                Create Test
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Optimization Results Modal */}
      <Dialog open={showOptimizationModal} onOpenChange={setShowOptimizationModal}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Optimization Results</DialogTitle>
          </DialogHeader>
          
          <div className="space-y-4">
            {optimizationResults ? (
              <div>
                <div className="text-sm text-gray-600 mb-4">
                  Found {optimizationResults.optimizations_found} potential optimizations,
                  applied {optimizationResults.optimizations_applied}
                </div>
                
                {Object.entries(optimizationResults.results || {}).map(([param, result]: [string, any]) => (
                  <div key={param} className="border rounded p-3">
                    <div className="font-medium">{param}</div>
                    <div className="text-sm text-gray-600 mt-1">
                      Current: {result.current_value} → Optimal: {result.optimal_value}
                    </div>
                    <div className="text-sm text-green-600">
                      Expected improvement: {(result.expected_improvement * 100).toFixed(2)}%
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900 mx-auto"></div>
                <div className="mt-2 text-gray-600">Running optimization...</div>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default ConfigurationManager;
