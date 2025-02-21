import mongoose, { Schema } from 'mongoose';

const timetableSchema = new Schema({
    day: {
        type: String,
        enum: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        required: true
    },
    periods: [
        {
            periodNumber: { type: Number, required: true },
            subject: { type: String, required: true },
            class: { type: String, required: true },
            teacher: { type: Schema.Types.ObjectId, ref: 'User', required: true },
            isSubstituted: { type: Boolean, default: false },
            substitute: { type: Schema.Types.ObjectId, ref: 'User', default: null }
        }
    ]
});

export const Timetable = mongoose.model('Timetable', timetableSchema);
