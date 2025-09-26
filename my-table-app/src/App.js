import React, { useState } from 'react';
import './App.css';

const App = () => {
  // Sample JSON data
  const [data] = useState([
    { id: 1, name: 'John Doe', email: 'john@example.com', age: 28, department: 'Engineering' },
    { id: 2, name: 'Jane Smith', email: 'jane@example.com', age: 32, department: 'Marketing' },
    { id: 3, name: 'Mike Johnson', email: 'mike@example.com', age: 25, department: 'Sales' },
    { id: 4, name: 'Sarah Wilson', email: 'sarah@example.com', age: 29, department: 'Engineering' },
    { id: 5, name: 'Tom Brown', email: 'tom@example.com', age: 35, department: 'HR' },
  ]);

  const [filter, setFilter] = useState('');
  const [departmentFilter, setDepartmentFilter] = useState('');

  // Filter data based on search terms
  const filteredData = data.filter(item => {
    const matchesName = item.name.toLowerCase().includes(filter.toLowerCase());
    const matchesEmail = item.email.toLowerCase().includes(filter.toLowerCase());
    const matchesDepartment = departmentFilter === '' || item.department === departmentFilter;
    
    return (matchesName || matchesEmail) && matchesDepartment;
  });

  // Get unique departments for filter dropdown
  const departments = [...new Set(data.map(item => item.department))];

  return (
    <div className="App">
      <header className="App-header">
        <h1>Employee Table with Filters</h1>
        
        {/* Filters */}
        <div className="filters">
          <input
            type="text"
            placeholder="Search by name or email..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="search-input"
          />
          
          <select
            value={departmentFilter}
            onChange={(e) => setDepartmentFilter(e.target.value)}
            className="department-filter"
          >
            <option value="">All Departments</option>
            {departments.map(dept => (
              <option key={dept} value={dept}>{dept}</option>
            ))}
          </select>
        </div>

        {/* Table */}
        <table className="data-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Name</th>
              <th>Email</th>
              <th>Age</th>
              <th>Department</th>
            </tr>
          </thead>
          <tbody>
            {filteredData.map(item => (
              <tr key={item.id}>
                <td>{item.id}</td>
                <td>{item.name}</td>
                <td>{item.email}</td>
                <td>{item.age}</td>
                <td>{item.department}</td>
              </tr>
            ))}
          </tbody>
        </table>
        
        {filteredData.length === 0 && (
          <p>No results found matching your criteria.</p>
        )}
      </header>
    </div>
  );
};

export default App;