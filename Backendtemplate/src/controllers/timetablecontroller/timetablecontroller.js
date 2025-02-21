import { Timetable } from '../../models/Timetable.models.js';
import { User } from '../../models/user.models.js';

// Fetch timetable
export const getTimetable = async (req, res) => {
    try {
        const timetable = await Timetable.find().populate('periods.teacher').populate('periods.substitute');
        res.json(timetable);
    } catch (error) {
        res.status(500).json({ message: 'Error fetching timetable', error });
    }
};

// Find available teachers for substitution
export const getAvailableTeachers = async (req, res) => {
    const { day, period, subject } = req.body;

    try {
        // Find teachers not scheduled in the given period
        const busyTeachers = await Timetable.findOne({ day, 'periods.periodNumber': period })
            .select('periods.teacher')
            .lean();

        const busyTeacherIds = busyTeachers?.periods.map(p => p.teacher) || [];

        // Find teachers who are not busy and can teach the subject
        const availableTeachers = await User.find({
            role: 'Teacher',
            _id: { $nin: busyTeacherIds }
        });

        res.json(availableTeachers);
    } catch (error) {
        res.status(500).json({ message: 'Error fetching available teachers', error });
    }
};

// Substitute a teacher
export const substituteTeacher = async (req, res) => {
    const { day, periodNumber, originalTeacherId, substituteTeacherId } = req.body;

    try {
        const timetable = await Timetable.findOneAndUpdate(
            { day, 'periods.periodNumber': periodNumber, 'periods.teacher': originalTeacherId },
            {
                $set: {
                    'periods.$.isSubstituted': true,
                    'periods.$.substitute': substituteTeacherId
                }
            },
            { new: true }
        ).populate('periods.teacher').populate('periods.substitute');

        if (!timetable) {
            return res.status(404).json({ message: 'Timetable entry not found' });
        }

        res.json({ message: 'Substitution successful', timetable });
    } catch (error) {
        res.status(500).json({ message: 'Error substituting teacher', error });
    }
};
