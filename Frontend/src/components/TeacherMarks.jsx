import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Select from 'react-select';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css'; 
import DatePicker from 'react-datepicker';
import "react-datepicker/dist/react-datepicker.css";

const TeacherMarks = () => {
  // Core state
  const [selectedClass, setSelectedClass] = useState(null);
  const [selectedExamType, setSelectedExamType] = useState('');
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submittingRecords, setSubmittingRecords] = useState({});
  const [selectedDate, setSelectedDate] = useState(new Date());

  // Data state with better structure
  const [marksState, setMarksState] = useState({
    // Structure: { studentId: { subjectName: { value: number, status: 'new'|'submitted'|'edited'|'editing', originalValue: number } } }
    marksData: {}
  });
  const months = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];

  const [examTypes] = useState(['Unit Test', 'Mid Term', 'Final Term']);
  const [subjectsByStudent, setSubjectsByStudent] = useState({});

  // Create axios instance
  const axiosInstance = axios.create({
    baseURL: 'http://localhost:8080/api/v1',
    withCredentials: true,
    headers: { 'Content-Type': 'application/json' }
  });

  // Generate class options
  const classOptions = Array.from({ length: 20 }, (_, i) => {
    const grade = Math.floor(i / 2) + 1;
    const section = i % 2 === 0 ? 'A' : 'B';
    const value = `${grade}${section}`;
    return { value, label: `Class ${value}` };
  });

  // Initialize marks data structure when students change
  useEffect(() => {
    if (students.length > 0) {
      const initialMarksData = {};
      
      students.forEach(student => {
        initialMarksData[student._id] = {};
        
        // If we have subjects for this student, initialize them
        if (subjectsByStudent[student._id]) {
          subjectsByStudent[student._id].forEach(subject => {
            initialMarksData[student._id][subject] = { 
              value: '', 
              status: 'new',
              originalValue: ''
            };
          });
        }
      });
      
      setMarksState(prev => ({
        ...prev,
        marksData: {
          ...prev.marksData,
          ...initialMarksData
        }
      }));
    }
  }, [students, subjectsByStudent]);

  // Error handling function
  const handleError = (error) => {
    console.error('Error details:', error);
    const errorMessage = error.response?.data?.message || error.message || 'An error occurred';
    toast.error(errorMessage);

    if (error.response?.status === 401) {
      toast.error('Session expired. Please login again');
    }
  };

  // Fetch subjects for a specific student
  const fetchStudentSubjects = async (studentId) => {
    try {
      const response = await axiosInstance.get(`/students/getEnrolledSubjects/${studentId}`);
      return response.data.subjects;
    } catch (error) {
      console.error('Error fetching subjects:', error);
      handleError(error);
      return [];
    }
  };

  // Fetch students for a class and their subjects
  const fetchStudents = async (className) => {
    setLoading(true);
    try {
      const response = await axiosInstance.post('/students/getStudentListByClass', { class: className });
      
      if (response.data.studentList) {
        const studentsData = response.data.studentList;
        setStudents(studentsData);
        
        // Fetch subjects for all students in parallel
        const subjectsPromises = studentsData.map(async (student) => {
          const subjects = await fetchStudentSubjects(student._id);
          return { [student._id]: subjects };
        });
        
        const subjectsResults = await Promise.all(subjectsPromises);
        const combinedSubjects = Object.assign({}, ...subjectsResults);
        setSubjectsByStudent(combinedSubjects);
      }
    } catch (error) {
      handleError(error);
    } finally {
      setLoading(false);
    }
  };

  // Handle class selection change
  const handleClassChange = (selectedOption) => {
    setSelectedClass(selectedOption);
    // Reset marks state when class changes
    setMarksState({ marksData: {} });
    if (selectedOption?.value) {
      fetchStudents(selectedOption.value);
    }
  };

  // Handle marks input change with validation
  const handleMarksChange = (studentId, subject, value) => {
    // Empty value is allowed
    if (value === '') {
      updateMarkValue(studentId, subject, value);
      return;
    }
    
    // Validate number format and range
    const numValue = Number(value);
    if (isNaN(numValue)) {
      toast.error('Please enter a valid number');
      return;
    }
    
    if (numValue < 0 || numValue > 100) {
      toast.error('Marks must be between 0 and 100');
      return;
    }
    
    updateMarkValue(studentId, subject, value);
  };

  // Update mark value and set appropriate status
  const updateMarkValue = (studentId, subject, value) => {
    setMarksState(prev => {
      const currentMark = prev.marksData[studentId]?.[subject];
      let newStatus = 'new';
      
      if (currentMark?.status === 'submitted') {
        newStatus = 'edited';
      } else if (currentMark?.status === 'editing') {
        newStatus = 'editing';
      }
      
      return {
        ...prev,
        marksData: {
          ...prev.marksData,
          [studentId]: {
            ...prev.marksData[studentId],
            [subject]: { 
              value,
              status: newStatus,
              originalValue: currentMark?.originalValue || ''
            }
          }
        }
      };
    });
  };

  // Start editing a submitted record
  const handleEditRecord = (studentId, subject) => {
    setMarksState(prev => {
      const currentMark = prev.marksData[studentId]?.[subject];
      if (!currentMark) return prev;
      
      return {
        ...prev,
        marksData: {
          ...prev.marksData,
          [studentId]: {
            ...prev.marksData[studentId],
            [subject]: { 
              ...currentMark,
              status: 'editing',
              originalValue: currentMark.value
            }
          }
        }
      };
    });
  };

  // Cancel editing and revert to original value
  const handleCancelEdit = (studentId, subject) => {
    setMarksState(prev => {
      const currentMark = prev.marksData[studentId]?.[subject];
      if (!currentMark) return prev;
      
      return {
        ...prev,
        marksData: {
          ...prev.marksData,
          [studentId]: {
            ...prev.marksData[studentId],
            [subject]: { 
              value: currentMark.originalValue,
              status: 'submitted',
              originalValue: currentMark.originalValue
            }
          }
        }
      };
    });
  };

  // Validate a single mark record
  const validateMark = (studentId, subject) => {
    const markData = marksState.marksData[studentId]?.[subject];
    if (!markData || markData.value === '' || (markData.status !== 'new' && markData.status !== 'edited' && markData.status !== 'editing')) {
      return null;
    }
    
    const numValue = Number(markData.value);
    if (isNaN(numValue) || numValue < 0 || numValue > 100) {
      const student = students.find(s => s._id === studentId);
      toast.error(`Invalid marks for ${student?.name || 'Student'} in ${subject}: ${markData.value}`);
      return null;
    }
    
    return {
      studentId,
      subject,
      marks: numValue
    };
  };

  // Submit a single mark record
  const handleSubmitRecord = async (studentId, subject) => {
    // Validation checks
    const validationChecks = [
      { condition: !selectedExamType, message: 'Please select an exam type' },
      { condition: !selectedClass, message: 'Please select a class' },
      { condition: !selectedDate, message: 'Please select a date' }
    ];
  
    // Check all validations
    for (const check of validationChecks) {
      if (check.condition) {
        toast.error(check.message);
        return;
      }
    }
      
    // Validate the mark record
    const markToSubmit = validateMark(studentId, subject);
    if (!markToSubmit) return; // Validation failed
      
    // Set loading state for this record
    const recordKey = `${studentId}-${subject}`;
    setSubmittingRecords(prev => ({ 
      ...prev, 
      [recordKey]: true 
    }));
      
    try {
      const student = students.find(s => s._id === studentId);
      
      const payload = {
        studentId,
        roll_no: student.roll_no,
        subjectName: subject,
        Exam_type: selectedExamType,
        marks: markToSubmit.marks,
        class: selectedClass.value,
        date: selectedDate
      };
        
      await axiosInstance.post('/students/addMarks', payload);
      toast.success(`Successfully submitted marks for ${student.name} in ${subject}`);
        
      // Update mark status to 'submitted'
      setMarksState(prev => {
        const updatedMarksData = {...prev.marksData};
          
        if (updatedMarksData[studentId]?.[subject]) {
          updatedMarksData[studentId][subject] = {
            value: markToSubmit.marks.toString(),
            status: 'submitted',
            originalValue: markToSubmit.marks.toString()
          };
        }
          
        return {
          ...prev,
          marksData: updatedMarksData
        };
      });
        
    } catch (error) {
      handleError(error);
    } finally {
      setSubmittingRecords(prev => ({ 
        ...prev, 
        [recordKey]: false 
      }));
    }
  };

  // Validate all marks before submission
  const validateAllMarks = () => {
    const marksToSubmit = [];
    const errors = [];
    
    Object.entries(marksState.marksData).forEach(([studentId, subjectMarks]) => {
      Object.entries(subjectMarks).forEach(([subject, markData]) => {
        // Skip empty or already submitted marks
        if (markData.value === '' || markData.status === 'submitted') return;
        
        const numValue = Number(markData.value);
        if (isNaN(numValue) || numValue < 0 || numValue > 100) {
          const student = students.find(s => s._id === studentId);
          errors.push(`Invalid marks for ${student?.name || 'Student'} in ${subject}: ${markData.value}`);
        } else {
          marksToSubmit.push({
            studentId, 
            subject, 
            marks: numValue
          });
        }
      });
    });
    
    if (errors.length > 0) {
      errors.forEach(error => toast.error(error));
      return null;
    }
    
    return marksToSubmit;
  };

  // Submit all marks
  const handleSubmitAllMarks = async () => {
    // Validation checks
    if (!selectedDate) {
      toast.error('Please select a date');
      return;
    }

    if (!selectedExamType) {
      toast.error('Please select an exam type');
      return;
    }

    if (!selectedClass) {
      toast.error('Please select a class');
      return;
    }
    
    // Validate marks and get submission payload
    const marksToSubmit = validateAllMarks();
    if (!marksToSubmit) return; // Validation failed
    
    if (marksToSubmit.length === 0) {
      toast.info('No new marks to submit');
      return;
    }
    
    // Submit marks
    setSubmitting(true);
    try {
      const promises = marksToSubmit.map(({studentId, subject, marks}) => {
        const student = students.find(s => s._id === studentId);
        const payload = {
          studentId,
          roll_no: student.roll_no,
          subjectName: subject,
          Exam_type: selectedExamType,
          marks,
          class: selectedClass.value,
          date: selectedDate // Added date to payload
        };
        
        return axiosInstance.post('/students/addMarks', payload);
      });
      
      await Promise.all(promises);
      toast.success(`Successfully submitted marks for ${marksToSubmit.length} entries`);
      
      // Update marks status to 'submitted'
      setMarksState(prev => {
        const updatedMarksData = {...prev.marksData};
        
        marksToSubmit.forEach(({studentId, subject, marks}) => {
          if (updatedMarksData[studentId]?.[subject]) {
            updatedMarksData[studentId][subject] = {
              value: marks.toString(),
              status: 'submitted',
              originalValue: marks.toString()
            };
          }
        });
        
        return {
          ...prev,
          marksData: updatedMarksData
        };
      });
      
    } catch (error) {
      handleError(error);
    } finally {
      setSubmitting(false);
    }
  };

  // Reset all marks
  const resetAllMarks = () => {
    // Confirm reset
    if (window.confirm('Are you sure you want to reset all marks? This will clear all entered data.')) {
      setMarksState({ marksData: {} });
      // Re-initialize the form by fetching students again
      if (selectedClass?.value) {
        fetchStudents(selectedClass.value);
      }
      toast.info('All marks have been reset');
    }
  };

  // Get styling for input field based on mark status
  const getInputStyle = (studentId, subject) => {
    const markData = marksState.marksData[studentId]?.[subject];
    if (!markData) return '';
    
    switch (markData.status) {
      case 'submitted':
        return 'bg-green-50 border-green-200';
      case 'edited':
        return 'bg-yellow-50 border-yellow-200';
      case 'editing':
        return 'bg-orange-50 border-orange-200';
      case 'new':
        return markData.value ? 'bg-blue-50 border-blue-200' : '';
      default:
        return '';
    }
  };

  // Count marks by status
  const getMarksCounts = () => {
    const counts = { new: 0, edited: 0, submitted: 0, editing: 0 };
    
    Object.values(marksState.marksData).forEach(subjectMarks => {
      Object.values(subjectMarks).forEach(markData => {
        if (markData.value !== '') {
          counts[markData.status]++;
        }
      });
    });
    
    return counts;
  };

  const marksCounts = getMarksCounts();

  return (
    <div className="container mx-auto px-4 py-8">
      <ToastContainer position="top-right" />
      
      <div className="bg-white rounded-lg shadow-md p-6">
        <h1 className="text-2xl font-bold mb-6">Enter Student Marks</h1>

        <div className="flex flex-wrap gap-4 mb-6">
          <div className="w-64">
            <label className="block text-sm font-medium text-gray-700 mb-1">Class</label>
            <Select
              value={selectedClass}
              onChange={handleClassChange}
              options={classOptions}
              placeholder="Select Class"
              isDisabled={loading || submitting}
              className="basic-select"
              classNamePrefix="select"
            />
          </div>

          <div className="w-64">
            <label className="block text-sm font-medium text-gray-700 mb-1">Exam Type</label>
            <select
              value={selectedExamType}
              onChange={(e) => setSelectedExamType(e.target.value)}
              className="w-full border border-gray-300 rounded-md p-2"
              disabled={loading || submitting}
            >
              <option value="">Select Exam Type</option>
              {examTypes.map(type => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
          </div>
          <div className="w-64">
  <label className="block text-sm font-medium text-gray-700 mb-1">Date</label>
  <DatePicker
    selected={selectedDate}
    onChange={date => setSelectedDate(date)}
    className="w-full border border-gray-300 rounded-md p-2"
    dateFormat="MMMM d, yyyy"
    maxDate={new Date()} // This prevents selection of future dates
    disabled={loading || submitting}
    placeholderText="Select date"
  />
</div>
        </div>

        {loading && (
          <div className="flex justify-center items-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          </div>
        )}

        {students.length > 0 && (
          <>
            <div className="flex flex-wrap justify-between mb-4 gap-2">
              <div className="flex flex-wrap gap-2">
                <span className="inline-block px-2 py-1 bg-blue-50 border border-blue-200 text-xs rounded-md">
                  New Entries: {marksCounts.new}
                </span>
                <span className="inline-block px-2 py-1 bg-yellow-50 border border-yellow-200 text-xs rounded-md">
                  Modified Entries: {marksCounts.edited}
                </span>
                <span className="inline-block px-2 py-1 bg-orange-50 border border-orange-200 text-xs rounded-md">
                  Editing: {marksCounts.editing}
                </span>
                <span className="inline-block px-2 py-1 bg-green-50 border border-green-200 text-xs rounded-md">
                  Submitted Entries: {marksCounts.submitted}
                </span>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={resetAllMarks}
                  className="bg-red-50 text-red-600 border border-red-200 hover:bg-red-100 px-4 py-2 rounded transition"
                  disabled={submitting}
                >
                  Reset All
                </button>
                <button
                  onClick={handleSubmitAllMarks}
                  className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded transition flex items-center"
                  disabled={submitting || (marksCounts.new + marksCounts.edited === 0)}
                >
                  {submitting ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Submitting...
                    </>
                  ) : (
                    `Submit All Marks (${marksCounts.new + marksCounts.edited})`
                  )}
                </button>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="min-w-full bg-white border border-gray-200">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="px-6 py-3 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Roll No.
                    </th>
                    <th className="px-6 py-3 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Name
                    </th>
                    {students[0] && subjectsByStudent[students[0]._id]?.map(subject => (
                      <th key={subject} className="px-6 py-3 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        {subject}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {students.map((student) => (
                    <tr key={student._id} className="hover:bg-gray-50 transition-colors">
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {student.roll_no}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {student.name}
                      </td>
                      {subjectsByStudent[student._id]?.map(subject => (
                        <td key={subject} className="px-6 py-4 whitespace-nowrap text-sm">
                          <div className="flex items-center gap-2">
                            <input
                              type="number"
                              min="0"
                              max="100"
                              className={`w-20 border rounded px-2 py-1 focus:ring-2 focus:ring-blue-300 focus:border-blue-300 transition-colors ${getInputStyle(student._id, subject)}`}
                              value={marksState.marksData[student._id]?.[subject]?.value || ''}
                              onChange={(e) => handleMarksChange(student._id, subject, e.target.value)}
                              placeholder="0-100"
                              disabled={submitting || submittingRecords[`${student._id}-${subject}`] || marksState.marksData[student._id]?.[subject]?.status === 'submitted'}
                            />
                            
                            {/* Action buttons based on status */}
                            {marksState.marksData[student._id]?.[subject]?.status === 'submitted' && (
                              <button
                                className="text-blue-600 hover:text-blue-800"
                                onClick={() => handleEditRecord(student._id, subject)}
                                disabled={submitting}
                                title="Edit"
                              >
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                                </svg>
                              </button>
                            )}
                            
                            {(marksState.marksData[student._id]?.[subject]?.status === 'new' || 
                              marksState.marksData[student._id]?.[subject]?.status === 'edited') && 
                              marksState.marksData[student._id]?.[subject]?.value && (
                              <button
                                className="text-green-600 hover:text-green-800"
                                onClick={() => handleSubmitRecord(student._id, subject)}
                                disabled={submitting || submittingRecords[`${student._id}-${subject}`]}
                                title="Submit"
                              >
                                {submittingRecords[`${student._id}-${subject}`] ? (
                                  <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                  </svg>
                                ) : (
                                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                  </svg>
                                )}
                              </button>
                            )}
                            
                            {marksState.marksData[student._id]?.[subject]?.status === 'editing' && (
                              <>
                                <button
                                  className="text-green-600 hover:text-green-800"
                                  onClick={() => handleSubmitRecord(student._id, subject)}
                                  disabled={submitting || submittingRecords[`${student._id}-${subject}`]}
                                  title="Save"
                                >
                                  {submittingRecords[`${student._id}-${subject}`] ? (
                                    <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                  ) : (
                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                    </svg>
                                  )}
                                </button>
                                <button
                                  className="text-red-600 hover:text-red-800"
                                  onClick={() => handleCancelEdit(student._id, subject)}
                                  disabled={submitting}
                                  title="Cancel Edit"
                                >
                                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                  </svg>
                                </button>
                              </>
                            )}
                          </div>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}

        {students.length === 0 && !loading && selectedClass && (
          <div className="text-center py-8 text-gray-500 bg-gray-50 rounded-md">
            <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <h3 className="mt-2 text-sm font-medium text-gray-900">No students found</h3>
            <p className="mt-1 text-sm text-gray-500">There are no students registered in this class.</p>
          </div>
        )}
        
        {!selectedClass && !loading && (
          <div className="bg-blue-50 border-l-4 border-blue-400 p-4 mb-6">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-blue-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-blue-700">
                  Please select a class to view students and enter marks.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TeacherMarks;