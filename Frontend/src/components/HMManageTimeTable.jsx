import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './TimeTable.css';

const HMManageTimeTable = () => {
  // State for timetable data
  const [timetable, setTimetable] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedCell, setSelectedCell] = useState(null);
  const [availableTeachers, setAvailableTeachers] = useState([]);
  const [showSubstitutionModal, setShowSubstitutionModal] = useState(false);
  
  // Define days and periods
  const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'];
  const periods = [1, 2, 3, 4, 5, 6, 7, 8];

  // Fetch timetable data on component mount
  useEffect(() => {
    const fetchTimetable = async () => {
      try {
        const res = await axios.get('/api/timetable');
        console.log("Timetable API response:", res.data);
        
        // Ensure timetable is an array
        if (Array.isArray(res.data)) {
          setTimetable(res.data);
        } else if (res.data && res.data.data && Array.isArray(res.data.data)) {
          // Handle nested response structure if needed
          setTimetable(res.data.data);
        } else {
          console.error("Unexpected timetable data format:", res.data);
          setTimetable([]);
          setError("Invalid data format received from server");
        }
        
        setLoading(false);
      } catch (err) {
        console.error("Error fetching timetable:", err);
        setError("Failed to load timetable data");
        setLoading(false);
      }
    };
    
    fetchTimetable();
  }, []);

  // Handle cell click for substitution
  const handleCellClick = async (day, periodNumber) => {
    // Find the period data for the clicked cell
    const periodData = Array.isArray(timetable) 
      ? timetable.find(t => t.day === day)?.periods.find(p => p.periodNumber === periodNumber)
      : null;
    
    if (!periodData) {
      console.log("No class scheduled for this period");
      return;
    }
    
    // Set the selected cell for substitution
    setSelectedCell({ day, periodNumber, periodData });
    
    try {
      // Fetch available teachers for substitution
      const res = await axios.post('/api/timetable/available-teachers', {
        day,
        period: periodNumber,
        subject: periodData.subject
      });
      
      setAvailableTeachers(res.data);
      setShowSubstitutionModal(true);
    } catch (err) {
      console.error("Error fetching available teachers:", err);
    }
  };

  // Handle teacher substitution
  const handleSubstitution = async (substituteTeacherId) => {
    if (!selectedCell) return;
    
    try {
      const { day, periodNumber, periodData } = selectedCell;
      
      await axios.put(`/api/timetable/substitute`, {
        day,
        periodNumber,
        originalTeacherId: periodData.teacher._id,
        substituteTeacherId
      });
      
      // Refresh timetable data after substitution
      const res = await axios.get('/api/timetable');
      setTimetable(Array.isArray(res.data) ? res.data : []);
      
      setShowSubstitutionModal(false);
      setSelectedCell(null);
    } catch (err) {
      console.error("Error making substitution:", err);
    }
  };

  // Close substitution modal
  const closeSubstitutionModal = () => {
    setShowSubstitutionModal(false);
    setSelectedCell(null);
  };

  if (loading) return <div className="loading-container">Loading timetable...</div>;
  if (error) return <div className="error-container">{error}</div>;

  return (
    <div className="timetable-management-container">
      <h2>Timetable Management</h2>
      
      <div className="timetable-container">
        <table className="timetable">
          <thead>
            <tr>
              <th>Period</th>
              {days.map(day => <th key={day}>{day}</th>)}
            </tr>
          </thead>
          <tbody>
            {periods.map(period => (
              <tr key={period}>
                <td className="period-number">{period}</td>
                {days.map(day => {
                  // Safely access the day data and period data
                  const dayData = Array.isArray(timetable) 
                    ? timetable.find(t => t.day === day) 
                    : null;
                  
                  const periodData = dayData?.periods?.find(p => p.periodNumber === period);
                  
                  return (
                    <td 
                      key={`${day}-${period}`}
                      className={`timetable-cell ${periodData?.isSubstituted ? 'substituted' : ''} ${!periodData ? 'free-period' : ''}`}
                      onClick={() => periodData && handleCellClick(day, period)}
                    >
                      {periodData ? (
                        <div className="cell-content">
                          <div className="subject">{periodData.subject}</div>
                          <div className="class-name">{periodData.class}</div>
                          <div className="teacher-name">
                            {periodData.isSubstituted 
                              ? <>
                                  <span className="original">Original: {periodData.teacher.name}</span>
                                  <span className="substitute">Sub: {periodData.substitute.name}</span>
                                </>
                              : periodData.teacher.name
                            }
                          </div>
                        </div>
                      ) : (
                        <div className="free-text">Free</div>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Substitution Modal */}
      {showSubstitutionModal && selectedCell && (
        <div className="modal-overlay">
          <div className="substitution-modal">
            <h3>Select Substitute Teacher</h3>
            <p>
              <strong>Day:</strong> {selectedCell.day}<br />
              <strong>Period:</strong> {selectedCell.periodNumber}<br />
              <strong>Subject:</strong> {selectedCell.periodData.subject}<br />
              <strong>Class:</strong> {selectedCell.periodData.class}<br />
              <strong>Original Teacher:</strong> {selectedCell.periodData.teacher.name}
            </p>
            
            {availableTeachers.length > 0 ? (
              <div className="available-teachers-list">
                <h4>Available Teachers:</h4>
                <ul>
                  {availableTeachers.map(teacher => (
                    <li key={teacher._id}>
                      <button 
                        className="substitute-button"
                        onClick={() => handleSubstitution(teacher._id)}
                      >
                        {teacher.name}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <p className="no-teachers-message">No available teachers for this period and subject.</p>
            )}
            
            <button className="close-modal-button" onClick={closeSubstitutionModal}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default HMManageTimeTable;