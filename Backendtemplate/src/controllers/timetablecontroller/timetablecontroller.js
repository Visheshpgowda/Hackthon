import Timetable from '../../models/timetable.models.js';
import { User } from '../../models/user.models.js';

// Fetch timetable

export const addTimetable = async (req, res) => {
    try {
        const { className, day, periods } = req.body;
        
        // Ensure each timetable entry includes a className, day, and periods
        if (!className || !day || !periods || periods.length !== 6) {
            return res.status(400).json({ error: "Invalid input. Ensure className, day, and exactly 6 periods are provided." });
        }

        // Create a new timetable entry
        const timetable = new Timetable({ className, day, periods });

        // Save to the database
        await timetable.save();

        res.status(201).json({ message: 'Timetable added successfully', timetable });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
};


export const getTimetable = async (req, res) => {
    try {
        const timetable = await Timetable.find().populate('periods.teacher', 'name email');
        res.json(timetable);
    } catch (error) {
        res.status(500).json({ error: error.message });
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
