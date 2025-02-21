import mongoose from 'mongoose';

const PeriodSchema = new mongoose.Schema({
    periodNumber: { type: Number, required: true },
    time: { type: String, required: true },
    subject: { type: String, required: true },
    class: { type: String, required: true },
    teacher: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
    isSubstituted: { type: Boolean, default: false },
    substituteTeacher: { type: mongoose.Schema.Types.ObjectId, ref: 'User', default: null }
});

const TimetableSchema = new mongoose.Schema({
    day: { type: String, required: true },
    periods: [PeriodSchema]
});

export default mongoose.model('Timetable', TimetableSchema);
