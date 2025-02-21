import { Marks } from "../../models/Marks.models.js";
export const addMarks = async (req, res) => {
    try {
      const { studentId, roll_no, subjectName, Exam_type: examType, marks, class: className } = req.body;
      
      const teacherId = req.user._id;
        console.log(studentId, roll_no, subjectName, examType, marks, className, teacherId);
      console.log(req.body);

        
      // Validate required fields
      if (!studentId || !roll_no || !subjectName || !examType || !marks || !className) {
        return res.status(400).json({
          success: false,
          message: 'All fields are required'
        });
      }
  
      // Validate marks range
      if (marks < 0 || marks > 100) {
        return res.status(400).json({
          success: false,
          message: 'Marks must be between 0 and 100'
        });
      }
  
      // Create new marks entry
      const marksEntry = new Marks({
        studentId,
        roll_no,
        subjectName,
        examType,
        marks,
        class: className,
        teacherId
      });
  
      await marksEntry.save();
  
      res.status(201).json({
        success: true,
        message: 'Marks added successfully',
        data: marksEntry
      });
  
    } catch (error) {
      console.error('Error in addMarks:', error);
      res.status(500).json({
        success: false,
        message: error.message || 'Error while adding marks'
      });
    }
  };

export const getMarksByRollNo = async(req, res) => {
    try {
        const roll_no = req.params.roll_no;
        const marks = await Marks.find({ roll_no });
        res.status(200).json({ marks });
    } catch (error) {
        res.status(404).json({ message: error.message });
    }
}

export const getMarksBySubject = async(req, res) => {
    try {
        const Subject_name = req.params.Subject_name;
        const marks = await Marks.find({ Subject_name });
        res.status(200).json({ marks });
    } catch (error) {
        res.status(404).json({ message: error.message });
    }
}

export const studentPerformance = async(req, res) => {
    try {
        const roll_no = req.params.roll_no;

        // Fetch all marks for the student with populated teacher data
        const marks = await Marks.find({ roll_no })
            .populate('teacher', 'name')
            .populate('Subject_name', 'Subject_name')
            .lean();

        if (marks.length === 0) {
            return res.status(404).json({ message: "No marks data found for this student" });
        }

        // Basic information
        const studentRoll = marks[0]?.roll_no || "Unknown";
        const totalEntries = marks.length;
        const uniqueSubjects = [...new Set(marks.map(m => m.Subject_name?._id.toString()))];
        const uniqueTeachers = [...new Set(marks.map(m => m.teacher?._id.toString()))];

        // Overall marks statistics
        const allMarks = marks.map(mark => mark.marks);
        const avgMark = (allMarks.reduce((sum, mark) => sum + mark, 0) / allMarks.length).toFixed(2);
        const highestMark = Math.max(...allMarks);
        const lowestMark = Math.min(...allMarks);
        const totalMarks = allMarks.reduce((sum, mark) => sum + mark, 0);
        const medianMark = calculateMedian(allMarks);

        // Pass/Fail Statistics (assuming pass mark is 50)
        const passCount = marks.filter(mark => mark.marks >= 50).length;
        const failCount = marks.filter(mark => mark.marks < 50).length;
        const passRate = ((passCount / totalEntries) * 100).toFixed(2);

        // Grade distribution
        const gradeRanges = {
            'A+': { min: 90, max: 100, count: 0 },
            'A': { min: 80, max: 89.99, count: 0 },
            'B+': { min: 75, max: 79.99, count: 0 },
            'B': { min: 70, max: 74.99, count: 0 },
            'C+': { min: 65, max: 69.99, count: 0 },
            'C': { min: 60, max: 64.99, count: 0 },
            'D+': { min: 55, max: 59.99, count: 0 },
            'D': { min: 50, max: 54.99, count: 0 },
            'F': { min: 0, max: 49.99, count: 0 }
        };

        // Count grades
        marks.forEach(mark => {
            for (const [grade, range] of Object.entries(gradeRanges)) {
                if (mark.marks >= range.min && mark.marks <= range.max) {
                    range.count++;
                    break;
                }
            }
        });

        // Calculate grade distribution
        const gradeDistribution = {};
        Object.entries(gradeRanges).forEach(([grade, range]) => {
            gradeDistribution[grade] = {
                count: range.count,
                percentage: ((range.count / totalEntries) * 100).toFixed(2)
            };
        });

        // Subject-wise performance
        const subjectPerformance = {};
        uniqueSubjects.forEach(subjectId => {
            const subjectMarks = marks.filter(mark => mark.Subject_name?._id.toString() === subjectId);
            const subjectName = subjectMarks[0]?.Subject_name?.Subject_name || "Unknown Subject";
            const subjectAvg = subjectMarks.reduce((sum, mark) => sum + mark.marks, 0) / subjectMarks.length;
            const highestMark = Math.max(...subjectMarks.map(mark => mark.marks));
            const lowestMark = Math.min(...subjectMarks.map(mark => mark.marks));
            const passStatus = subjectMarks.every(mark => mark.marks >= 50) ? 'Pass' : 'Fail';

            // Get best and worst exam in this subject
            const sortedExams = [...subjectMarks].sort((a, b) => b.marks - a.marks);
            const bestExam = sortedExams[0];
            const worstExam = sortedExams[sortedExams.length - 1];

            subjectPerformance[subjectName] = {
                averageMark: subjectAvg.toFixed(2),
                highestMark,
                lowestMark,
                examCount: subjectMarks.length,
                passStatus,
                bestExam: {
                    marks: bestExam.marks,
                    examType: bestExam.Exam_type,
                    teacher: bestExam.teacher?.name || 'Unknown'
                },
                worstExam: {
                    marks: worstExam.marks,
                    examType: worstExam.Exam_type,
                    teacher: worstExam.teacher?.name || 'Unknown'
                }
            };
        });

        // Performance comparison by exam type
        const midtermMarks = marks.filter(mark => mark.Exam_type === 'Midterm');
        const finalMarks = marks.filter(mark => mark.Exam_type === 'Final');

        const examTypeComparison = {
            Midterm: midtermMarks.length > 0 ? {
                count: midtermMarks.length,
                averageMark: (midtermMarks.reduce((sum, mark) => sum + mark.marks, 0) / midtermMarks.length).toFixed(2),
                highestMark: Math.max(...midtermMarks.map(m => m.marks)),
                lowestMark: Math.min(...midtermMarks.map(m => m.marks)),
                passRate: ((midtermMarks.filter(m => m.marks >= 50).length / midtermMarks.length) * 100).toFixed(2)
            } : null,

            Final: finalMarks.length > 0 ? {
                count: finalMarks.length,
                averageMark: (finalMarks.reduce((sum, mark) => sum + mark.marks, 0) / finalMarks.length).toFixed(2),
                highestMark: Math.max(...finalMarks.map(m => m.marks)),
                lowestMark: Math.min(...finalMarks.map(m => m.marks)),
                passRate: ((finalMarks.filter(m => m.marks >= 50).length / finalMarks.length) * 100).toFixed(2)
            } : null
        };

        // Calculate improvement/decline between midterm and final
        const subjectProgression = {};
        uniqueSubjects.forEach(subjectId => {
            const subjectMarks = marks.filter(mark => mark.Subject_name?._id.toString() === subjectId);
            const subjectName = subjectMarks[0]?.Subject_name?.Subject_name || "Unknown Subject";
            const midtermForSubject = subjectMarks.filter(m => m.Exam_type === 'Midterm');
            const finalForSubject = subjectMarks.filter(m => m.Exam_type === 'Final');

            if (midtermForSubject.length > 0 && finalForSubject.length > 0) {
                const midtermAvg = midtermForSubject.reduce((sum, m) => sum + m.marks, 0) / midtermForSubject.length;
                const finalAvg = finalForSubject.reduce((sum, m) => sum + m.marks, 0) / finalForSubject.length;
                const improvement = finalAvg - midtermAvg;

                subjectProgression[subjectName] = {
                    midtermAverage: midtermAvg.toFixed(2),
                    finalAverage: finalAvg.toFixed(2),
                    improvement: improvement.toFixed(2),
                    improvementPercentage: ((improvement / midtermAvg) * 100).toFixed(2),
                    trend: improvement > 0 ? 'Improving' : improvement < 0 ? 'Declining' : 'Stable'
                };
            }
        });

        // Ranking data (best to worst subjects)
        const subjectRankings = Object.entries(subjectPerformance)
            .map(([subject, data]) => ({
                subject,
                averageMark: parseFloat(data.averageMark)
            }))
            .sort((a, b) => b.averageMark - a.averageMark)
            .map((item, index) => ({
                ...item,
                rank: index + 1
            }));

        // Performance trend over time
        const marksByDate = marks.sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt))
            .map(mark => ({
                subject: mark.Subject_name?.Subject_name || "Unknown Subject",
                examType: mark.Exam_type,
                marks: mark.marks,
                date: mark.createdAt
            }));

        // Strength and weakness analysis
        const strengthSubject = subjectRankings[0];
        const weaknessSubject = subjectRankings[subjectRankings.length - 1];

        // Overall performance categorization
        let performanceCategory;
        if (parseFloat(avgMark) >= 80) {
            performanceCategory = "Excellent";
        } else if (parseFloat(avgMark) >= 70) {
            performanceCategory = "Good";
        } else if (parseFloat(avgMark) >= 60) {
            performanceCategory = "Average";
        } else if (parseFloat(avgMark) >= 50) {
            performanceCategory = "Below Average";
        } else {
            performanceCategory = "Needs Improvement";
        }

        // Performance suggestions based on data
        const suggestions = [];

        // Check for failed subjects
        const failedSubjects = Object.entries(subjectPerformance)
            .filter(([_, data]) => data.passStatus === 'Fail')
            .map(([subject]) => subject);

        if (failedSubjects.length > 0) {
            suggestions.push(`Focus on improving in: ${failedSubjects.join(', ')}`);
        }

        // Check for decline between midterm and final
        const decliningSubjects = Object.entries(subjectProgression)
            .filter(([_, data]) => data.trend === 'Declining')
            .map(([subject]) => subject);

        if (decliningSubjects.length > 0) {
            suggestions.push(`Performance is declining in: ${decliningSubjects.join(', ')}. Additional support may be needed.`);
        }

        // Check for inconsistency
        const inconsistentSubjects = Object.entries(subjectPerformance)
            .filter(([_, data]) => Math.abs(data.highestMark - data.lowestMark) > 30)
            .map(([subject]) => subject);

        if (inconsistentSubjects.length > 0) {
            suggestions.push(`Performance is inconsistent in: ${inconsistentSubjects.join(', ')}. More consistent study habits may help.`);
        }

        // Response object with all statistics
        const response = {
            studentInfo: {
                roll_no,
                performanceCategory,
                overallAverage: avgMark,
                totalExams: totalEntries
            },
            overallStats: {
                totalMarks,
                averageMark: avgMark,
                medianMark,
                highestMark,
                lowestMark,
                passRate,
                failRate: (100 - parseFloat(passRate)).toFixed(2)
            },
            gradeDistribution,
            subjectPerformance,
            subjectRankings,
            strengthsAndWeaknesses: {
                strongestSubject: strengthSubject,
                weakestSubject: weaknessSubject
            },
            examTypeComparison,
            progressionAnalysis: subjectProgression,
            suggestions,
            performanceTrend: marksByDate,
            rawData: marks.map(mark => ({
                subject: mark.Subject_name?.Subject_name || "Unknown Subject",
                examType: mark.Exam_type,
                marks: mark.marks,
                teacher: mark.teacher?.name || 'Unknown',
                date: mark.createdAt
            }))
        };

        res.status(200).json(response);
    } catch (error) {
        res.status(500).json({
            message: "Error analyzing student performance",
            error: error.message
        });
    }
};

// Helper function to calculate median
function calculateMedian(arr) {
    const sorted = [...arr].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);

    if (sorted.length % 2 === 0) {
        return ((sorted[middle - 1] + sorted[middle]) / 2).toFixed(2);
    }

    return sorted[middle].toFixed(2);
}