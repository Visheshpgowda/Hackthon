import { Marks } from "../../models/Marks.models";

export const teacherPerformance = async(req, res) => {
    try {
        const teacherId = req.params.teacher;

        // Fetch all marks for the teacher with populated student data
        const marks = await Marks.find({ teacher: teacherId })
            .populate('Student_name', 'name')
            .populate('teacher', 'name')
            .lean();

        if (marks.length === 0) {
            return res.status(404).json({ message: "No marks data found for this teacher" });
        }

        // Basic information
        const teacherName = marks[0] ? .teacher ? .name || "Unknown Teacher";
        const totalEntries = marks.length;
        const uniqueStudents = new Set(marks.map(m => m.Student_name ? ._id.toString())).size;
        const uniqueSubjects = [...new Set(marks.map(m => m.subject))];

        // Grade distribution (assuming marks are out of 100)
        // Define grade ranges
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

        // Calculate grade percentages
        const gradeDistribution = {};
        Object.entries(gradeRanges).forEach(([grade, range]) => {
            gradeDistribution[grade] = {
                count: range.count,
                percentage: ((range.count / totalEntries) * 100).toFixed(2)
            };
        });

        // Pass/Fail Statistics (assuming pass mark is 50)
        const passCount = marks.filter(mark => mark.marks >= 50).length;
        const failCount = marks.filter(mark => mark.marks < 50).length;
        const passRate = ((passCount / totalEntries) * 100).toFixed(2);
        const failRate = ((failCount / totalEntries) * 100).toFixed(2);

        // Performance categories
        const excellentCount = marks.filter(mark => mark.marks >= 80).length;
        const goodCount = marks.filter(mark => mark.marks >= 70 && mark.marks < 80).length;
        const averageCount = marks.filter(mark => mark.marks >= 60 && mark.marks < 70).length;
        const belowAverageCount = marks.filter(mark => mark.marks >= 50 && mark.marks < 60).length;
        const poorCount = marks.filter(mark => mark.marks < 50).length;

        const performanceCategories = {
            excellent: {
                count: excellentCount,
                percentage: ((excellentCount / totalEntries) * 100).toFixed(2)
            },
            good: {
                count: goodCount,
                percentage: ((goodCount / totalEntries) * 100).toFixed(2)
            },
            average: {
                count: averageCount,
                percentage: ((averageCount / totalEntries) * 100).toFixed(2)
            },
            belowAverage: {
                count: belowAverageCount,
                percentage: ((belowAverageCount / totalEntries) * 100).toFixed(2)
            },
            poor: {
                count: poorCount,
                percentage: ((poorCount / totalEntries) * 100).toFixed(2)
            }
        };

        // Subject-wise performance
        const subjectPerformance = {};
        uniqueSubjects.forEach(subject => {
            const subjectMarks = marks.filter(mark => mark.subject === subject);
            const subjectAvg = subjectMarks.reduce((sum, mark) => sum + mark.marks, 0) / subjectMarks.length;
            const highestMark = Math.max(...subjectMarks.map(mark => mark.marks));
            const lowestMark = Math.min(...subjectMarks.map(mark => mark.marks));
            const subjectPassRate = ((subjectMarks.filter(mark => mark.marks >= 50).length / subjectMarks.length) * 100).toFixed(2);

            // Calculate standard deviation
            const mean = subjectAvg;
            const squareDiffs = subjectMarks.map(mark => {
                const diff = mark.marks - mean;
                return diff * diff;
            });
            const avgSquareDiff = squareDiffs.reduce((sum, diff) => sum + diff, 0) / squareDiffs.length;
            const stdDev = Math.sqrt(avgSquareDiff).toFixed(2);

            subjectPerformance[subject] = {
                count: subjectMarks.length,
                averageMark: subjectAvg.toFixed(2),
                highestMark,
                lowestMark,
                passRate: subjectPassRate,
                standardDeviation: stdDev
            };
        });

        // Exam type comparison (Midterm vs Final)
        const midtermMarks = marks.filter(mark => mark.Exam_type === 'Midterm');
        const finalMarks = marks.filter(mark => mark.Exam_type === 'Final');

        const examTypeComparison = {
            Midterm: {
                count: midtermMarks.length,
                averageMark: midtermMarks.length > 0 ?
                    (midtermMarks.reduce((sum, mark) => sum + mark.marks, 0) / midtermMarks.length).toFixed(2) :
                    0,
                passRate: midtermMarks.length > 0 ?
                    ((midtermMarks.filter(mark => mark.marks >= 50).length / midtermMarks.length) * 100).toFixed(2) :
                    0
            },
            Final: {
                count: finalMarks.length,
                averageMark: finalMarks.length > 0 ?
                    (finalMarks.reduce((sum, mark) => sum + mark.marks, 0) / finalMarks.length).toFixed(2) :
                    0,
                passRate: finalMarks.length > 0 ?
                    ((finalMarks.filter(mark => mark.marks >= 50).length / finalMarks.length) * 100).toFixed(2) :
                    0
            }
        };

        // Top and bottom performers
        const studentPerformance = {};
        marks.forEach(mark => {
            const studentId = mark.Student_name ? ._id.toString();
            if (!studentPerformance[studentId]) {
                studentPerformance[studentId] = {
                    studentId,
                    studentName: mark.Student_name ? .name || `Student (${mark.roll_no})`,
                    rollNo: mark.roll_no,
                    marks: [],
                    totalMarks: 0,
                    examCount: 0
                };
            }

            studentPerformance[studentId].marks.push(mark.marks);
            studentPerformance[studentId].totalMarks += mark.marks;
            studentPerformance[studentId].examCount += 1;
        });

        // Calculate average for each student
        Object.values(studentPerformance).forEach(student => {
            student.averageMark = (student.totalMarks / student.examCount).toFixed(2);
        });

        // Sort students by average mark
        const sortedStudents = Object.values(studentPerformance)
            .sort((a, b) => b.averageMark - a.averageMark);

        const topPerformers = sortedStudents.slice(0, 5);
        const bottomPerformers = sortedStudents.slice(-5).reverse();

        // Calculate overall statistics
        const allMarks = marks.map(mark => mark.marks);
        const avgMark = (allMarks.reduce((sum, mark) => sum + mark, 0) / allMarks.length).toFixed(2);
        const highestMark = Math.max(...allMarks);
        const lowestMark = Math.min(...allMarks);
        const medianMark = calculateMedian(allMarks);

        // Calculate quartiles
        const sortedMarks = [...allMarks].sort((a, b) => a - b);
        const q1 = calculateMedian(sortedMarks.slice(0, Math.floor(sortedMarks.length / 2)));
        const q3 = calculateMedian(sortedMarks.slice(Math.ceil(sortedMarks.length / 2)));
        const iqr = q3 - q1;

        // Response object with all statistics
        const response = {
            teacherInfo: {
                teacherId,
                teacherName
            },
            overallStats: {
                totalEntries,
                uniqueStudents,
                uniqueSubjects: uniqueSubjects.length,
                averageMark: avgMark,
                medianMark,
                highestMark,
                lowestMark,
                passRate,
                failRate,
                quartiles: {
                    q1,
                    q2: medianMark,
                    q3,
                    iqr
                }
            },
            performanceCategories,
            gradeDistribution,
            subjectPerformance,
            examTypeComparison,
            topPerformers,
            bottomPerformers,
            // Include raw data for custom frontend processing if needed
            rawData: marks.map(mark => ({
                studentName: mark.Student_name ? .name || 'Unknown',
                rollNo: mark.roll_no,
                subject: mark.subject,
                examType: mark.Exam_type,
                marks: mark.marks
            }))
        };

        res.status(200).json(response);
    } catch (error) {
        res.status(500).json({
            message: "Error analyzing teacher performance",
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