import React, { useState, useEffect } from 'react';

const AdminPage = () => {
  const [documents, setDocuments] = useState([]);
  const [schemaTree, setSchemaTree] = useState({});
  
  // 파일 처리 대기열
  // item 구조: { id: string, file: File, status: 'idle'|'analyzing'|'ready'|'saving'|'saved'|'error', result: object, error: string }
  const [uploadQueue, setUploadQueue] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [clientId] = useState(() => Math.random().toString(36).substr(2, 9));

  const API_BASE = "http://localhost:8000/api/admin";
  const WS_BASE = "ws://localhost:8000/ws";

  // 1. 문서 목록 조회
  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_BASE}/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data);
      }
    } catch (err) {
      console.error("Failed to fetch documents", err);
    }
  };

  // 1-2. 스키마 트리 조회
  const fetchSchema = async () => {
    try {
      const res = await fetch(`${API_BASE}/schema`);
      if (res.ok) {
        const data = await res.json();
        setSchemaTree(data);
      }
    } catch (err) {
      console.error("Failed to fetch schema", err);
    }
  };

  useEffect(() => {
    fetchDocuments();
    fetchSchema();
  }, []);

  // WebSocket 연결
  useEffect(() => {
    const ws = new WebSocket(`${WS_BASE}/${clientId}`);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'progress') {
        setUploadQueue(prev => prev.map(item => 
          item.id === data.item_id 
          ? { ...item, progress: data.progress, statusText: data.message } 
          : item
        ));
      }
    };

    return () => ws.close();
  }, [clientId]);

  // 2. 파일 추가 (드래그 앤 드롭 & 선택)
  const addFiles = (files) => {
    const newItems = Array.from(files).map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      file,
      status: 'idle',
      result: null,
      error: null,
      progress: 0
    }));
    setUploadQueue(prev => [...prev, ...newItems]);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(e.dataTransfer.files);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      addFiles(e.target.files);
    }
  };

  // 3. 개별 파일 분석
  const analyzeItem = async (itemId) => {
    const item = uploadQueue.find(i => i.id === itemId);
    if (!item) return;

    updateItemStatus(itemId, 'analyzing');
    const formData = new FormData();
    formData.append("file", item.file);
    formData.append("client_id", clientId);
    formData.append("item_id", itemId);

    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Analysis failed"); 
      const data = await res.json();
      updateItemStatus(itemId, 'ready', { result: data });
    } catch (err) {
      updateItemStatus(itemId, 'error', { error: err.message });
    }
  };

  // 4. 스키마만 업데이트 (데이터 적재 X)
  const updateSchemaItem = async (itemId) => {
    const item = uploadQueue.find(i => i.id === itemId);
    if (!item || !item.result) return;

    updateItemStatus(itemId, 'saving', { statusText: 'Updating Schema...' });

    try {
      const res = await fetch(`${API_BASE}/schema/update`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          entities: item.result.entities,
          relations: item.result.relations 
        }),
      });
      if (!res.ok) throw new Error("Schema update failed");

      updateItemStatus(itemId, 'saved', { statusText: 'Schema Updated' });
      fetchSchema(); // 스키마 트리 갱신

      // [UI/UX] Remove saved item from queue after a delay
      setTimeout(() => {
        setUploadQueue(prev => prev.filter(item => item.id !== itemId));
      }, 2000);

    } catch (err) {
      updateItemStatus(itemId, 'error', { error: err.message });
    }
  };

  // 5. 데이터 저장 (스키마 + 데이터 적재)
  const saveItem = async (itemId) => {
    const item = uploadQueue.find(i => i.id === itemId);
    if (!item || !item.result) return;

    updateItemStatus(itemId, 'saving', { statusText: 'Saving Data...' });

    try {
      const res = await fetch(`${API_BASE}/save`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(item.result),
      });
      if (!res.ok) throw new Error("Save failed");

      updateItemStatus(itemId, 'saved', { statusText: 'Saved to DB' });
      fetchDocuments(); // 문서 목록 갱신
      fetchSchema(); // 스키마 트리 갱신

      // [UI/UX] Remove saved item from queue after a delay
      setTimeout(() => {
        setUploadQueue(prev => prev.filter(item => item.id !== itemId));
      }, 2000);

    } catch (err) {
      updateItemStatus(itemId, 'error', { error: err.message });
    }
  };

  // 상태 업데이트 헬퍼
  const updateItemStatus = (id, status, extra = {}) => {
    setUploadQueue(prev => prev.map(item => 
      item.id === id ? { ...item, status, ...extra } : item
    ));
  };

  // 일괄 분석 시작
  const analyzeAll = () => {
    uploadQueue.forEach(item => {
      if (item.status === 'idle') analyzeItem(item.id);
    });
  };

  // 5. 문서 삭제
  const handleDelete = async (docId) => {
    if (!window.confirm("Are you sure you want to delete this document?")) return;

    // [UI/UX] Optimistic Update: 즉시 UI에서 제거
    const originalDocuments = documents;
    setDocuments(prev => prev.filter(doc => doc.id !== docId));

    try {
      const res = await fetch(`${API_BASE}/documents/${docId}`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error('Deletion failed on server');
    } catch (err) {
      alert("Failed to delete document. Restoring list.");
      setDocuments(originalDocuments); // [Rollback] 실패 시 목록 복원
      alert("Failed to delete document");
    }
  };

  // 6. 지식 그래프 내보내기
  const handleExport = async () => {
    try {
      const res = await fetch(`${API_BASE}/export/json`);
      if (!res.ok) throw new Error("Export failed");
      
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `knowledge_graph_${new Date().toISOString().slice(0,10)}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      alert(`Export Error: ${err.message}`);
    }
  };

  // UI 헬퍼: 상태 뱃지
  const getStatusBadge = (status) => {
    const styles = {
      idle: { bg: '#e0e0e0', color: '#333', text: 'Waiting' },
      analyzing: { bg: '#fff3cd', color: '#856404', text: 'Analyzing' },
      ready: { bg: '#d4edda', color: '#155724', text: 'Ready' },
      saving: { bg: '#cce5ff', color: '#004085', text: 'Updating' },
      saved: { bg: '#28a745', color: '#fff', text: 'Done' },
      error: { bg: '#f8d7da', color: '#721c24', text: 'Failed' }
    };
    const s = styles[status] || styles.idle;
    return (
      <span style={{ 
        padding: '4px 8px', borderRadius: '12px', fontSize: '0.8em', 
        backgroundColor: s.bg, color: s.color, fontWeight: 'bold' 
      }}>
        {s.text}
      </span>
    );
  };

  // UI 헬퍼: 프로그레스 바
  const ProgressBar = ({ progress }) => {
    if (!progress) return null;
    return (
      <div style={{ width: '100%', backgroundColor: '#e0e0e0', borderRadius: '4px', marginTop: '5px', height: '6px' }}>
        <div style={{ width: `${progress}%`, backgroundColor: '#007bff', height: '100%', borderRadius: '4px', transition: 'width 0.3s' }}></div>
      </div>
    );
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h1>🛠️ Ontology Construction Mode</h1>
      <p style={{ color: '#666' }}>Focus on building the schema first. Data ingestion is disabled in this mode.</p>
      
      <div style={{ display: "flex", gap: "20px" }}>
        {/* 왼쪽: 스키마 뷰어 */}
        <div style={{ flex: 1, border: "1px solid #ccc", padding: "15px", borderRadius: "8px" }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
            <h3>🌳 Current Ontology</h3>
            <button onClick={handleExport} style={{ background: '#6c757d', color: 'white', border: 'none', padding: '5px 10px', borderRadius: '4px', cursor: 'pointer', fontSize: '0.9em' }}>
              📥 Export JSON
            </button>
          </div>
          {Object.keys(schemaTree).length === 0 ? <p>No schema defined yet.</p> : (
            <div style={{ maxHeight: '600px', overflowY: 'auto' }}>
              {Object.entries(schemaTree).map(([parent, children]) => {
                if (parent === 'relations') return null; // 관계는 별도 처리
                return (
                  <div key={parent} style={{ marginBottom: '15px' }}>
                    <div style={{ fontWeight: 'bold', color: '#0056b3', marginBottom: '5px' }}>{parent} (L2)</div>
                    <div style={{ paddingLeft: '15px', borderLeft: '2px solid #eee' }}>
                      {children.map(child => (
                        <div key={child} style={{ padding: '2px 0', color: '#333' }}>• {child}</div>
                      ))}
                    </div>
                  </div>
                );
              })}
              {schemaTree.relations && (
                <div>
                  <div style={{ fontWeight: 'bold', color: '#28a745', marginBottom: '5px', marginTop: '15px', borderTop: '1px solid #ccc', paddingTop: '15px' }}>🔗 Relations</div>
                  <div style={{ paddingLeft: '15px' }}>
                    {schemaTree.relations.map(rel => (
                      <div key={rel} style={{ padding: '2px 0', color: '#333' }}>• {rel}</div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          <hr style={{ margin: '20px 0' }} />
          
          {/* 문서 목록 (참고용) */}
          <h3>📂 Uploaded Documents</h3>
          {documents.length === 0 ? <p>No documents found.</p> : (
            <ul style={{ listStyle: "none", padding: 0 }}>
              {documents.map((doc) => (
                <li key={doc.id} style={{ marginBottom: "10px", padding: "10px", background: "#f9f9f9", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div>
                    <strong>{doc.name}</strong>
                    <br />
                    <small style={{ color: "#666" }}>{doc.date}</small>
                  </div>
                  <button 
                    onClick={() => handleDelete(doc.id)}
                    style={{ background: "#ff4d4d", color: "white", border: "none", padding: "5px 10px", cursor: "pointer", borderRadius: "4px" }}
                  >
                    Delete
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* 오른쪽: 업로드 및 분석 */}
        <div style={{ flex: 2, border: "1px solid #ccc", padding: "15px", borderRadius: "8px" }}>
          <h3>🚀 Upload & Analyze</h3>
          
          {/* 드래그 앤 드롭 영역 */}
          <div 
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            style={{
              border: `2px dashed ${isDragging ? '#4CAF50' : '#ccc'}`,
              borderRadius: '8px',
              padding: '30px',
              textAlign: 'center',
              backgroundColor: isDragging ? '#f0fff4' : '#fafafa',
              marginBottom: '20px',
              transition: 'all 0.2s'
            }}
          >
            <p style={{ margin: 0, color: '#666' }}>
              Drag & Drop files here or <br/>
              <label style={{ color: '#007bff', cursor: 'pointer', textDecoration: 'underline' }}>
                browse files
                <input type="file" multiple onChange={handleFileSelect} style={{ display: 'none' }} />
              </label>
            </p>
          </div>

          {/* 대기열 목록 */}
          {uploadQueue.length > 0 && (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <h4>Queue ({uploadQueue.length})</h4>
                <button onClick={analyzeAll} style={{ padding: '5px 15px', cursor: 'pointer', background: '#007bff', color: 'white', border: 'none', borderRadius: '4px' }}>
                  Analyze All Idle
                </button>
              </div>

              <ul style={{ listStyle: 'none', padding: 0 }}>
                {uploadQueue.map(item => (
                  <li key={item.id} style={{ background: 'white', border: '1px solid #eee', marginBottom: '8px', padding: '10px', borderRadius: '4px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ flex: 1, marginRight: '10px', overflow: 'hidden' }}>
                      <div style={{ fontWeight: 'bold' }}>{item.file.name}</div>
                      <div style={{ fontSize: '0.85em', color: '#666', marginTop: '4px' }}>
                        {getStatusBadge(item.status)}
                        {item.result && <span style={{ marginLeft: '10px' }}>Found: {Object.keys(item.result.entities).length} types</span>}
                        {item.error && <span style={{ marginLeft: '10px', color: 'red' }}>{item.error}</span>}
                      </div>
                      {item.status === 'analyzing' && <ProgressBar progress={item.progress} />}
                    </div>
                    <div>
                      {item.status === 'idle' && (
                        <button onClick={() => analyzeItem(item.id)} style={{ marginRight: '5px', cursor: 'pointer' }}>Analyze</button>
                      )}
                      {item.status === 'ready' && (
                        <div style={{ display: 'flex', gap: '5px' }}>
                          <button onClick={() => updateSchemaItem(item.id)} style={{ background: '#17a2b8', color: 'white', border: 'none', padding: "5px 10px", borderRadius: "4px", cursor: "pointer" }} title="Only define types, do not insert data">Update Schema Only</button>
                          <button onClick={() => saveItem(item.id)} style={{ background: '#28a745', color: 'white', border: 'none', padding: "5px 10px", borderRadius: "4px", cursor: "pointer" }} title="Save schema and insert data">Save Data</button>
                        </div>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AdminPage;